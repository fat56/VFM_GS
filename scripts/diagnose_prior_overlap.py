#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_feature(path: Path, storage: str) -> np.ndarray:
    if storage in ("npy_float32", "npy_float16"):
        value = np.load(path)
    elif storage == "npz_uint8":
        with np.load(path) as archive:
            value = archive["feature"].astype(np.float32) / 255.0
    else:
        raise ValueError("Unsupported cache storage {!r}".format(storage))
    return np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize01(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if value.size == 0:
        return value
    value_min = float(value.min())
    value_max = float(value.max())
    return (value - value_min) / max(value_max - value_min, eps)


def _resize_map(value: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if value.shape[:2] == (height, width):
        return value.astype(np.float32)
    image = Image.fromarray(np.clip(_normalize01(value) * 255.0, 0, 255).astype(np.uint8))
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dx = np.zeros_like(luma, dtype=np.float32)
    dy = np.zeros_like(luma, dtype=np.float32)
    dx[:, 1:] = luma[:, 1:] - luma[:, :-1]
    dy[1:, :] = luma[1:, :] - luma[:-1, :]
    return _normalize01(np.sqrt(dx * dx + dy * dy + 1e-12))


def _topk_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    fraction = min(max(float(fraction), 0.0), 1.0)
    flat = value.reshape(-1)
    mask = np.zeros(flat.shape, dtype=bool)
    if fraction <= 0.0 or flat.size == 0:
        return mask.reshape(value.shape)
    positive = np.count_nonzero(flat > 0)
    if positive == 0:
        return mask.reshape(value.shape)
    k = min(max(1, int(math.ceil(flat.size * fraction))), int(positive))
    indices = np.argpartition(flat, -k)[-k:]
    mask[indices] = True
    return mask.reshape(value.shape)


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(value[mask]))


def _safe_ratio(num: float, denom: float) -> float:
    if abs(denom) < 1e-12:
        return 0.0
    return float(num / denom)


def _infer_method_dir(model_dir: Path, split: str, method: str | None) -> Path:
    split_dir = model_dir / split
    if method:
        method_dir = split_dir / method
        if not method_dir.exists():
            raise FileNotFoundError("Missing method directory {}".format(method_dir))
        return method_dir
    candidates = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            "Unable to infer method under {}. Pass --method. Candidates: {}".format(
                split_dir, ", ".join(path.name for path in candidates)
            )
        )
    return candidates[0]


def _load_render_index(model_dir: Path) -> dict[str, str]:
    cameras_path = model_dir / "cameras.json"
    with cameras_path.open("r", encoding="utf-8") as handle:
        cameras = json.load(handle)
    return {
        "{:05d}.png".format(idx): str(camera["img_name"])
        for idx, camera in enumerate(cameras)
    }


def _load_cache(cache_dir: Path) -> tuple[dict, dict]:
    with (cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Cache manifest has no entries: {}".format(cache_dir))
    return manifest, entries


def _load_prior_map(
    image_name: str,
    cache_dir: Path,
    manifest: dict,
    entries: dict,
    size_hw: tuple[int, int],
) -> np.ndarray:
    if image_name not in entries:
        raise KeyError("Image {!r} is missing from {}".format(image_name, cache_dir))
    entry = entries[image_name]
    storage = entry.get("storage", manifest.get("storage", "npy_float32"))
    feature = _load_feature(cache_dir / entry["cache_file"], storage)
    if feature.ndim == 3:
        # Descriptor caches are not directly comparable as a prior map. For quick
        # diagnostics, use channel magnitude as a coarse saliency proxy.
        feature = np.linalg.norm(feature.astype(np.float32), axis=-1)
    if feature.ndim != 2:
        raise ValueError("Expected 2D prior map or 3D descriptor feature, got {}".format(feature.shape))
    return _resize_map(_normalize01(feature), size_hw)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.floating)) and key not in {"view_index"}
    ]
    summary = {"num_views": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = float(values.mean())
    return summary


def diagnose(args: argparse.Namespace) -> dict:
    baseline_dir = Path(args.baseline_model)
    candidate_dir = Path(args.candidate_model) if args.candidate_model else None
    baseline_method_dir = _infer_method_dir(baseline_dir, args.split, args.method)
    candidate_method_dir = _infer_method_dir(candidate_dir, args.split, args.method) if candidate_dir else None
    render_to_image = _load_render_index(baseline_dir)
    manifest, entries = _load_cache(Path(args.prior_cache))

    render_dir = baseline_method_dir / "renders"
    gt_dir = baseline_method_dir / "gt"
    candidate_render_dir = candidate_method_dir / "renders" if candidate_method_dir else None
    rows = []
    topk = float(args.topk)
    rgb_topk = float(args.rgb_topk)

    for render_path in sorted(render_dir.iterdir()):
        if render_path.suffix not in IMAGE_SUFFIXES:
            continue
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            raise FileNotFoundError("Missing GT image {}".format(gt_path))
        if render_path.name not in render_to_image:
            raise KeyError("Missing render index {} in cameras.json".format(render_path.name))

        render = _load_rgb(render_path)
        gt = _load_rgb(gt_path)
        rgb_error = np.mean(np.abs(render - gt), axis=-1).astype(np.float32)
        gt_edge = _edge_map(gt)
        image_name = render_to_image[render_path.name]
        prior = _load_prior_map(image_name, Path(args.prior_cache), manifest, entries, rgb_error.shape)

        prior_mask = _topk_mask(prior, topk)
        rgb_mask = _topk_mask(rgb_error, rgb_topk)
        gt_edge_mask = _topk_mask(gt_edge, topk)
        non_prior_mask = ~prior_mask

        row = {
            "view_index": int(Path(render_path.name).stem),
            "render_name": render_path.name,
            "image_name": image_name,
            "baseline_l1": float(np.mean(rgb_error)),
            "baseline_l1_prior_topk": _masked_mean(rgb_error, prior_mask),
            "baseline_l1_non_prior": _masked_mean(rgb_error, non_prior_mask),
            "prior_rgb_topk_iou": _safe_ratio(
                float(np.count_nonzero(prior_mask & rgb_mask)),
                float(np.count_nonzero(prior_mask | rgb_mask)),
            ),
            "prior_rgb_topk_recall": _safe_ratio(
                float(np.count_nonzero(prior_mask & rgb_mask)),
                float(np.count_nonzero(rgb_mask)),
            ),
            "prior_gt_edge_topk_iou": _safe_ratio(
                float(np.count_nonzero(prior_mask & gt_edge_mask)),
                float(np.count_nonzero(prior_mask | gt_edge_mask)),
            ),
            "prior_coverage": float(np.mean(prior_mask)),
            "rgb_error_topk_coverage": float(np.mean(rgb_mask)),
        }

        if candidate_render_dir is not None:
            candidate_path = candidate_render_dir / render_path.name
            if not candidate_path.exists():
                raise FileNotFoundError("Missing candidate render {}".format(candidate_path))
            candidate = _load_rgb(candidate_path)
            candidate_error = np.mean(np.abs(candidate - gt), axis=-1).astype(np.float32)
            improvement = rgb_error - candidate_error
            row.update(
                {
                    "candidate_l1": float(np.mean(candidate_error)),
                    "delta_l1": float(np.mean(candidate_error) - np.mean(rgb_error)),
                    "candidate_l1_prior_topk": _masked_mean(candidate_error, prior_mask),
                    "delta_l1_prior_topk": _masked_mean(candidate_error, prior_mask)
                    - _masked_mean(rgb_error, prior_mask),
                    "candidate_l1_non_prior": _masked_mean(candidate_error, non_prior_mask),
                    "delta_l1_non_prior": _masked_mean(candidate_error, non_prior_mask)
                    - _masked_mean(rgb_error, non_prior_mask),
                    "improvement_in_prior_topk_share": _safe_ratio(
                        float(np.sum(np.maximum(improvement, 0.0)[prior_mask])),
                        float(np.sum(np.maximum(improvement, 0.0))),
                    ),
                    "worse_in_prior_topk_share": _safe_ratio(
                        float(np.sum(np.maximum(-improvement, 0.0)[prior_mask])),
                        float(np.sum(np.maximum(-improvement, 0.0))),
                    ),
                }
            )
        rows.append(row)

    if not rows:
        raise ValueError("No render images found in {}".format(render_dir))

    summary = _summarize(rows)
    summary.update(
        {
            "baseline_model": str(baseline_dir),
            "candidate_model": str(candidate_dir) if candidate_dir else None,
            "prior_cache": str(args.prior_cache),
            "split": args.split,
            "method": baseline_method_dir.name,
            "topk": topk,
            "rgb_topk": rgb_topk,
            "prior_feature": manifest.get("feature"),
            "prior_backend": manifest.get("backend"),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_view.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether prior top-k regions match RGB error bottlenecks.")
    parser.add_argument("--baseline-model", required=True, help="Rendered baseline model directory.")
    parser.add_argument("--candidate-model", default=None, help="Optional rendered candidate model directory.")
    parser.add_argument("--prior-cache", required=True, help="VFM cache directory with manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for per_view.csv and summary.json.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default=None, help="Render method directory, e.g. ours_30000.")
    parser.add_argument("--topk", type=float, default=0.25, help="Prior top-k fraction.")
    parser.add_argument("--rgb-topk", type=float, default=0.25, help="RGB error top-k fraction.")
    args = parser.parse_args()
    summary = diagnose(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
