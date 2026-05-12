#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_dino_descriptor_residual import (
    IMAGE_SUFFIXES,
    _extract_dino_tokens_from_array,
    _infer_method_dir,
    _load_cache_tokens,
    _load_dinov2_model,
    _load_render_index,
    _load_rgb,
    _normalize01,
    _resize_map,
    _safe_ratio,
    _smooth_2d,
    _topk_mask,
)


def _label_fraction(value: float) -> str:
    return str(int(round(float(value) * 100.0)))


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Empty run name in {!r}".format(value))
        return name, Path(path)
    path = Path(value)
    return path.name, path


def _masked_metrics(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0, "l1": 0.0, "mse": 0.0, "psnr": 0.0}
    abs_error = np.mean(np.abs(render - gt), axis=-1)
    mse_error = np.mean((render - gt) ** 2, axis=-1)
    l1 = float(np.mean(abs_error[mask]))
    mse = float(np.mean(mse_error[mask]))
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    return {"count": count, "l1": l1, "mse": mse, "psnr": psnr}


def _add_region(
    regions: dict[str, np.ndarray],
    name: str,
    mask: np.ndarray,
) -> None:
    if name in regions:
        raise ValueError("Duplicate region {!r}".format(name))
    regions[name] = mask.astype(bool)


def _build_regions(
    dino_norm: np.ndarray,
    rgb_norm: np.ndarray,
    topks: list[float],
    broad_topks: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    regions: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float | int]] = {}

    all_mask = np.ones_like(rgb_norm, dtype=bool)
    _add_region(regions, "all", all_mask)

    for topk in topks:
        label = _label_fraction(topk)
        dino_mask = _topk_mask(dino_norm, topk)
        rgb_mask = _topk_mask(rgb_norm, topk)
        intersection = dino_mask & rgb_mask
        union = dino_mask | rgb_mask
        dino_only = dino_mask & ~rgb_mask
        rgb_only = rgb_mask & ~dino_mask

        _add_region(regions, "dino_top{}".format(label), dino_mask)
        _add_region(regions, "rgb_top{}".format(label), rgb_mask)
        _add_region(regions, "dino_rgb_intersection_top{}".format(label), intersection)
        _add_region(regions, "dino_rgb_union_top{}".format(label), union)
        _add_region(regions, "dino_only_top{}".format(label), dino_only)
        _add_region(regions, "rgb_only_top{}".format(label), rgb_only)

        stats["top{}".format(label)] = {
            "dino_count": int(np.count_nonzero(dino_mask)),
            "rgb_count": int(np.count_nonzero(rgb_mask)),
            "intersection_count": int(np.count_nonzero(intersection)),
            "union_count": int(np.count_nonzero(union)),
            "dino_only_count": int(np.count_nonzero(dino_only)),
            "rgb_only_count": int(np.count_nonzero(rgb_only)),
            "iou": _safe_ratio(float(np.count_nonzero(intersection)), float(np.count_nonzero(union))),
            "dino_recall_in_rgb": _safe_ratio(float(np.count_nonzero(intersection)), float(np.count_nonzero(dino_mask))),
            "rgb_recall_in_dino": _safe_ratio(float(np.count_nonzero(intersection)), float(np.count_nonzero(rgb_mask))),
        }

    for broad_topk in broad_topks:
        label = _label_fraction(broad_topk)
        _add_region(regions, "rgb_broad_top{}".format(label), _topk_mask(rgb_norm, broad_topk))

    return regions, stats


def _weighted_summary(per_view_rows: list[dict[str, str | int | float]], reference_name: str) -> dict:
    totals: dict[str, dict[str, dict[str, float]]] = {}
    for row in per_view_rows:
        run = str(row["run"])
        region = str(row["region"])
        count = float(row["count"])
        bucket = totals.setdefault(run, {}).setdefault(
            region,
            {"count": 0.0, "l1_sum": 0.0, "mse_sum": 0.0},
        )
        bucket["count"] += count
        bucket["l1_sum"] += float(row["l1"]) * count
        bucket["mse_sum"] += float(row["mse"]) * count

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for run, regions in totals.items():
        summary[run] = {}
        for region, values in regions.items():
            count = max(values["count"], 1.0)
            l1 = values["l1_sum"] / count
            mse = values["mse_sum"] / count
            summary[run][region] = {
                "count": int(values["count"]),
                "l1": float(l1),
                "mse": float(mse),
                "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
            }

    reference = summary.get(reference_name, {})
    for run, regions in summary.items():
        for region, values in regions.items():
            ref_values = reference.get(region)
            if not ref_values:
                continue
            values["delta_l1_vs_{}".format(reference_name)] = float(values["l1"] - ref_values["l1"])
            values["delta_mse_vs_{}".format(reference_name)] = float(values["mse"] - ref_values["mse"])
            values["delta_psnr_vs_{}".format(reference_name)] = float(values["psnr"] - ref_values["psnr"])
    return summary


def diagnose(args: argparse.Namespace) -> dict:
    reference_name, reference_dir = _parse_run(args.reference_run)
    runs = [_parse_run(value) for value in args.run]
    if not any(name == reference_name for name, _ in runs):
        runs.insert(0, (reference_name, reference_dir))

    reference_method = _infer_method_dir(reference_dir, args.split, args.method)
    reference_render_dir = reference_method / "renders"
    reference_gt_dir = reference_method / "gt"
    render_to_image = _load_render_index(reference_dir)

    gt_cache = Path(args.gt_cache)
    cache_manifest = None
    if gt_cache:
        cache_manifest = json.loads((gt_cache / "manifest.json").read_text(encoding="utf-8"))
        backend = cache_manifest.get("backend")
    else:
        backend = args.backend
    if not backend:
        raise ValueError("Unable to infer DINO backend.")

    model = _load_dinov2_model(backend, args.dinov2_repo, args.device, pretrained=True)

    run_methods = {
        name: _infer_method_dir(path, args.split, args.method)
        for name, path in runs
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_view_rows: list[dict[str, str | int | float]] = []
    mask_stat_totals: dict[str, dict[str, float]] = {}
    topks = [float(value) for value in args.topk]
    broad_topks = [float(value) for value in args.rgb_broad_topk]

    for render_path in sorted(reference_render_dir.iterdir()):
        if render_path.suffix not in IMAGE_SUFFIXES:
            continue
        gt_path = reference_gt_dir / render_path.name
        if not gt_path.exists():
            raise FileNotFoundError("Missing GT {}".format(gt_path))
        if render_path.name not in render_to_image:
            raise KeyError("Missing render index {} in cameras.json".format(render_path.name))

        image_name = render_to_image[render_path.name]
        reference_render = _load_rgb(render_path)
        gt = _load_rgb(gt_path)
        reference_rgb_error = np.mean(np.abs(reference_render - gt), axis=-1).astype(np.float32)

        gt_tokens_np, _, _ = _load_cache_tokens(gt_cache, image_name)
        gt_tokens = torch.from_numpy(gt_tokens_np).to(device=args.device, dtype=torch.float32)
        grid_size = tuple(gt_tokens.shape[:2])
        rendered_tokens = _extract_dino_tokens_from_array(reference_render, model, args.device, grid_size)
        gt_tokens = F.normalize(gt_tokens.to(torch.float32), dim=-1)
        patch_error = 0.5 * torch.clamp(
            1.0 - F.cosine_similarity(rendered_tokens, gt_tokens, dim=-1),
            min=0.0,
            max=2.0,
        )
        patch_error = _smooth_2d(patch_error, int(args.smooth_kernel))
        dino_pixel = _resize_map(
            patch_error.detach().cpu().numpy().astype(np.float32),
            reference_rgb_error.shape,
            args.upsample,
        )
        dino_norm = _normalize01(dino_pixel)
        rgb_norm = _normalize01(reference_rgb_error)
        regions, mask_stats = _build_regions(dino_norm, rgb_norm, topks, broad_topks)

        for label, values in mask_stats.items():
            total = mask_stat_totals.setdefault(
                label,
                {
                    "dino_count": 0.0,
                    "rgb_count": 0.0,
                    "intersection_count": 0.0,
                    "union_count": 0.0,
                    "dino_only_count": 0.0,
                    "rgb_only_count": 0.0,
                },
            )
            for key in total:
                total[key] += float(values[key])

        for run_name, method_dir in run_methods.items():
            run_render_path = method_dir / "renders" / render_path.name
            if not run_render_path.exists():
                raise FileNotFoundError("Missing render {}".format(run_render_path))
            run_render = _load_rgb(run_render_path)
            if run_render.shape != gt.shape:
                raise ValueError(
                    "Shape mismatch for {}: {} vs {}".format(run_render_path, run_render.shape, gt.shape)
                )
            for region_name, mask in regions.items():
                metrics = _masked_metrics(run_render, gt, mask)
                per_view_rows.append(
                    {
                        "view_index": int(render_path.stem),
                        "render_name": render_path.name,
                        "image_name": image_name,
                        "run": run_name,
                        "region": region_name,
                        **metrics,
                    }
                )

    if not per_view_rows:
        raise ValueError("No renders found in {}".format(reference_render_dir))

    for values in mask_stat_totals.values():
        values["iou"] = _safe_ratio(values["intersection_count"], values["union_count"])
        values["dino_recall_in_rgb"] = _safe_ratio(values["intersection_count"], values["dino_count"])
        values["rgb_recall_in_dino"] = _safe_ratio(values["intersection_count"], values["rgb_count"])

    summary = {
        "reference_run": reference_name,
        "reference_model": str(reference_dir),
        "runs": {name: str(path) for name, path in runs},
        "split": args.split,
        "method": reference_method.name,
        "gt_cache": str(gt_cache),
        "dinov2_repo": args.dinov2_repo,
        "device": args.device,
        "topk": topks,
        "rgb_broad_topk": broad_topks,
        "smooth_kernel": int(args.smooth_kernel),
        "upsample": args.upsample,
        "mask_stats": mask_stat_totals,
        "region_metrics": _weighted_summary(per_view_rows, reference_name),
    }

    with (output_dir / "per_view.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_view_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_view_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare 0003 runs on fixed RGB/DINO local regions from a reference render."
    )
    parser.add_argument("--reference-run", required=True, help="name=MODEL_DIR used to build masks.")
    parser.add_argument("--run", action="append", required=True, help="name=MODEL_DIR to evaluate; repeatable.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default=None)
    parser.add_argument("--gt-cache", default="output/0001/vfm_cache/bicycle_dinov2_vits14")
    parser.add_argument("--backend", default="dinov2_vits14")
    parser.add_argument("--dinov2-repo", default="output/0001/external/dinov2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=float, nargs="+", default=[0.25, 0.10])
    parser.add_argument("--rgb-broad-topk", type=float, nargs="+", default=[0.50])
    parser.add_argument("--smooth-kernel", type=int, default=3)
    parser.add_argument("--upsample", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = diagnose(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
