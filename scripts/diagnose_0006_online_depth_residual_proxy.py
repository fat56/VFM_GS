#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vfm_gs.gaussian_renderer import GaussianModel
from vfm_gs.gaussian_renderer import render_fastgs
from vfm_gs.scene import Scene
from vfm_gs.scorers.vfm_cache import load_feature, read_manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cfg_args(run_dir: Path) -> Namespace:
    cfg_path = run_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = eval(cfg_path.read_text(encoding="utf-8"), {"Namespace": Namespace})
    cfg.model_path = str(run_dir)
    return cfg


def select_views(views: list[Any], max_views: int, stride: int) -> list[Any]:
    stride = max(stride, 1)
    selected = list(views)[::stride]
    if max_views <= 0 or len(selected) <= max_views:
        return selected
    if max_views == 1:
        return [selected[0]]
    indexes = [
        round(index * (len(selected) - 1) / (max_views - 1))
        for index in range(max_views)
    ]
    return [selected[index] for index in indexes]


def normalize_valid(value: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = torch.zeros_like(value, dtype=torch.float32)
    valid_values = value[valid].to(torch.float32)
    if valid_values.numel() == 0:
        return out
    value_min = torch.min(valid_values)
    value_max = torch.max(valid_values)
    out[valid] = (value[valid].to(torch.float32) - value_min) / torch.clamp(value_max - value_min, min=eps)
    return torch.clamp(out, 0.0, 1.0)


def luma(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.sum(image[:3] * weights, dim=0)


def edge_map(image: torch.Tensor) -> torch.Tensor:
    gray = luma(image)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    dy[1:, :] = gray[1:, :] - gray[:-1, :]
    edge = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return normalize_valid(edge, torch.ones_like(edge, dtype=torch.bool))


def topk_mask(value: torch.Tensor, fraction: float, valid: torch.Tensor | None = None) -> torch.Tensor:
    flat = value.reshape(-1)
    if valid is None:
        valid_flat = torch.ones_like(flat, dtype=torch.bool)
    else:
        valid_flat = valid.reshape(-1)
    valid_indices = torch.nonzero(valid_flat, as_tuple=False).flatten()
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if valid_indices.numel() == 0:
        return mask.reshape_as(value)
    k = int(math.ceil(valid_indices.numel() * min(max(float(fraction), 0.0), 1.0)))
    if k <= 0:
        return mask.reshape_as(value)
    k = min(k, valid_indices.numel())
    valid_values = flat[valid_indices]
    top_indices = torch.topk(valid_values, k=k, largest=True).indices
    mask[valid_indices[top_indices]] = True
    return mask.reshape_as(value)


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def mask_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = torch.count_nonzero(left & right).item()
    union = torch.count_nonzero(left | right).item()
    return safe_ratio(float(intersection), float(union))


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    if torch.count_nonzero(mask).item() == 0:
        return 0.0
    return float(value[mask].mean().item())


def load_prior_map(cache_dir: Path, image_name: str, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    manifest = read_manifest(cache_dir)
    entries = manifest.get("entries", {})
    if image_name not in entries:
        raise KeyError("Image {!r} is missing from {}".format(image_name, cache_dir))
    entry = entries[image_name]
    storage = entry.get("storage", manifest.get("storage", "npy_float32"))
    feature = torch.from_numpy(load_feature(cache_dir / entry["cache_file"], storage)).to(device=device, dtype=torch.float32)
    if feature.ndim != 2:
        raise ValueError("Expected 2D depth prior for {!r}, got shape {}".format(image_name, list(feature.shape)))
    if tuple(feature.shape[-2:]) != tuple(size):
        feature = F.interpolate(
            feature.view(1, 1, *feature.shape[-2:]),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).view(*size)
    return normalize_valid(feature, torch.ones_like(feature, dtype=torch.bool))


def proxy_center_zbuffer_depth(
    gaussians: GaussianModel,
    viewpoint_cam: Any,
    chunk_size: int,
    splat_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = gaussians.get_xyz.detach()
    device = xyz.device
    height = int(viewpoint_cam.image_height)
    width = int(viewpoint_cam.image_width)
    depth_flat = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)

    for start in range(0, int(xyz.shape[0]), chunk_size):
        points = xyz[start : start + chunk_size]
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=device)
        points_h = torch.cat([points, ones], dim=1)
        view_points = torch.matmul(points_h, viewpoint_cam.world_view_transform)
        clip = torch.matmul(points_h, viewpoint_cam.full_proj_transform)
        clip_w = clip[:, 3]
        ndc = clip[:, :3] / torch.clamp(clip_w[:, None], min=1e-7)
        depth = view_points[:, 2].to(torch.float32)
        valid = (
            torch.isfinite(depth)
            & torch.isfinite(ndc).all(dim=1)
            & (clip_w > 0.0)
            & (depth > float(getattr(viewpoint_cam, "znear", 0.01)))
            & (ndc[:, 0] >= -1.0)
            & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0)
            & (ndc[:, 1] <= 1.0)
        )
        if not torch.any(valid):
            continue
        x = torch.clamp(((ndc[valid, 0] + 1.0) * 0.5 * width).to(torch.long), 0, width - 1)
        y = torch.clamp(((1.0 - ndc[valid, 1]) * 0.5 * height).to(torch.long), 0, height - 1)
        valid_depth = depth[valid]
        radius = max(0, int(splat_radius))
        for dy in range(-radius, radius + 1):
            yy = y + dy
            in_y = (yy >= 0) & (yy < height)
            if not torch.any(in_y):
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                in_bounds = in_y & (xx >= 0) & (xx < width)
                if not torch.any(in_bounds):
                    continue
                pixel_indices = yy[in_bounds] * width + xx[in_bounds]
                depth_flat.scatter_reduce_(
                    0,
                    pixel_indices,
                    valid_depth[in_bounds],
                    reduce="amin",
                    include_self=True,
                )

    valid_flat = torch.isfinite(depth_flat)
    return depth_flat.view(height, width), valid_flat.view(height, width)


def build_rows_by_scene(rows: list[dict[str, str]], method: str) -> dict[tuple[str, str], dict[str, str]]:
    selected = {}
    for row in rows:
        if row["method"] != method:
            continue
        selected[(row["dataset"], row["scene"])] = row
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["scene"]))].append(row)

    scene_rows = []
    for (dataset, scene), items in sorted(grouped.items()):
        scene_summary: dict[str, Any] = {
            "dataset": dataset,
            "scene": scene,
            "view_count": len(items),
        }
        for key, value in items[0].items():
            if isinstance(value, (float, int)) and key != "view_index":
                scene_summary[key] = sum(float(item[key]) for item in items) / len(items)
        scene_rows.append(scene_summary)

    overall: dict[str, Any] = {"scene_count": len(scene_rows), "view_count": len(rows)}
    if scene_rows:
        for key, value in scene_rows[0].items():
            if isinstance(value, (float, int)) and key != "view_count":
                overall[key] = sum(float(row[key]) for row in scene_rows) / len(scene_rows)
    return {"overall": overall, "scenes": scene_rows}


def diagnose_scene(row: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    run_dir = Path(row["run_dir"])
    cfg = load_cfg_args(run_dir)
    gaussians = GaussianModel(cfg.sh_degree, optimizer_type=getattr(cfg, "optimizer_type", "default"))
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = select_views(scene.getTrainCameras(), args.max_views, args.view_stride)
    cache_dir = Path(args.cache_template.format(dataset=row["dataset"], scene=row["scene"]))

    result_rows = []
    with torch.no_grad():
        for view_index, view in enumerate(views):
            rendered = torch.clamp(
                render_fastgs(view, gaussians, cfg, background, getattr(cfg, "mult", 0.5))["render"],
                0.0,
                1.0,
            )
            gt = torch.clamp(view.original_image[:3], 0.0, 1.0)
            rgb_error = torch.mean(torch.abs(rendered - gt), dim=0)
            gt_edges = edge_map(gt)

            depth, valid = proxy_center_zbuffer_depth(gaussians, view, args.chunk_size, args.splat_radius)
            valid_coverage = float(valid.to(torch.float32).mean().item())
            if torch.count_nonzero(valid).item() == 0:
                continue
            depth_norm = normalize_valid(depth, valid)
            inv_depth_norm = normalize_valid(1.0 / torch.clamp(depth, min=1e-6), valid)
            prior = load_prior_map(cache_dir, view.image_name, rgb_error.shape, rgb_error.device)
            residual_depth = torch.abs(depth_norm - prior)
            residual_inv_depth = torch.abs(inv_depth_norm - prior)

            rgb_mask = topk_mask(rgb_error, args.topk, valid)
            prior_mask = topk_mask(prior, args.topk, valid)
            residual_depth_mask = topk_mask(residual_depth, args.topk, valid)
            residual_inv_mask = topk_mask(residual_inv_depth, args.topk, valid)
            gt_edge_mask = topk_mask(gt_edges, args.topk, valid)
            valid_non_residual_inv = valid & ~residual_inv_mask

            result_rows.append(
                {
                    "dataset": row["dataset"],
                    "scene": row["scene"],
                    "method": row["method"],
                    "view_index": view_index,
                    "image_name": view.image_name,
                    "proxy_valid_coverage": valid_coverage,
                    "rgb_l1": masked_mean(rgb_error, valid),
                    "rgb_l1_prior_topk": masked_mean(rgb_error, prior_mask),
                    "rgb_l1_residual_depth_topk": masked_mean(rgb_error, residual_depth_mask),
                    "rgb_l1_residual_inv_topk": masked_mean(rgb_error, residual_inv_mask),
                    "rgb_l1_non_residual_inv": masked_mean(rgb_error, valid_non_residual_inv),
                    "prior_rgb_iou": mask_iou(prior_mask, rgb_mask),
                    "residual_depth_rgb_iou": mask_iou(residual_depth_mask, rgb_mask),
                    "residual_inv_rgb_iou": mask_iou(residual_inv_mask, rgb_mask),
                    "prior_gt_edge_iou": mask_iou(prior_mask, gt_edge_mask),
                    "residual_depth_gt_edge_iou": mask_iou(residual_depth_mask, gt_edge_mask),
                    "residual_inv_gt_edge_iou": mask_iou(residual_inv_mask, gt_edge_mask),
                    "mean_prior": masked_mean(prior, valid),
                    "mean_residual_depth": masked_mean(residual_depth, valid),
                    "mean_residual_inv_depth": masked_mean(residual_inv_depth, valid),
                    "run_dir": row["run_dir"],
                    "cache_dir": str(cache_dir),
                }
            )
    return result_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 0006 online depth residual proxy smoke.")
    parser.add_argument("--input-summary", type=Path, default=Path("output/0006/validation_selector/mipnerf360_depth_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/0006/online_depth_residual_proxy"))
    parser.add_argument("--cache-template", default="output/0002/vfm_cache/{scene}_depth_anything_v2s_depth")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--view-stride", type=int, default=17)
    parser.add_argument("--topk", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--splat-radius", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(torch.device("cuda:0"))
    datasets = set(args.datasets or [])
    scenes = set(args.scenes or [])
    rows_by_scene = build_rows_by_scene(read_csv(args.input_summary), args.method)
    selected_rows = [
        row
        for (dataset, scene), row in sorted(rows_by_scene.items())
        if (not datasets or dataset in datasets) and (not scenes or scene in scenes)
    ]
    if not selected_rows:
        raise ValueError("No rows selected for method={!r}".format(args.method))

    all_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        print("[{}:{}:{}] online residual proxy".format(row["dataset"], row["scene"], row["method"]), flush=True)
        all_rows.extend(diagnose_scene(row, args))
        write_csv(args.output_dir / "per_view.csv", all_rows)
        write_json(args.output_dir / "summary.json", summarize(all_rows))

    print("Wrote {}".format(args.output_dir / "per_view.csv"))
    print("Wrote {}".format(args.output_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
