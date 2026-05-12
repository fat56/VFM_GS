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
from PIL import Image

from vfm_gs.cli.build_vfm_cache import PATCH_SIZE, _image_to_dino_tensor, _load_dinov2_model, _resize_to_patch_grid
from vfm_gs.scorers.vfm_cache import load_feature, read_manifest


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _normalize01(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if value.size == 0:
        return value
    value_min = float(value.min())
    value_max = float(value.max())
    return (value - value_min) / max(value_max - value_min, eps)


def _topk_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    fraction = min(max(float(fraction), 0.0), 1.0)
    flat = value.reshape(-1)
    mask = np.zeros(flat.shape, dtype=bool)
    if fraction <= 0.0 or flat.size == 0:
        return mask.reshape(value.shape)
    positive = int(np.count_nonzero(flat > 0))
    if positive == 0:
        return mask.reshape(value.shape)
    k = min(max(1, int(math.ceil(flat.size * fraction))), positive)
    indices = np.argpartition(flat, -k)[-k:]
    mask[indices] = True
    return mask.reshape(value.shape)


def _safe_ratio(num: float, denom: float) -> float:
    if abs(denom) < 1e-12:
        return 0.0
    return float(num / denom)


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(value[mask]))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)
    br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = float(np.sqrt(np.sum(ar * ar) * np.sum(br * br)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(ar * br) / denom)


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
                split_dir,
                ", ".join(path.name for path in candidates),
            )
        )
    return candidates[0]


def _load_render_index(model_dir: Path) -> dict[str, str]:
    with (model_dir / "cameras.json").open("r", encoding="utf-8") as handle:
        cameras = json.load(handle)
    return {
        "{:05d}.png".format(idx): str(camera["img_name"])
        for idx, camera in enumerate(cameras)
    }


def _load_cache_tokens(cache_dir: Path, image_name: str) -> tuple[np.ndarray, dict, dict]:
    manifest = read_manifest(cache_dir)
    entries = manifest.get("entries", {})
    if image_name not in entries:
        raise KeyError("Image {!r} is missing from {}".format(image_name, cache_dir))
    entry = entries[image_name]
    storage = entry.get("storage", manifest.get("storage", "npy_float32"))
    tokens = load_feature(cache_dir / entry["cache_file"], storage)
    if tokens.ndim != 3:
        raise ValueError("Expected DINO descriptor cache [H,W,C], got {}".format(tokens.shape))
    return tokens.astype(np.float32), manifest, entry


def _extract_dino_tokens_from_image(
    image: Image.Image,
    model,
    device: str,
    max_width: int | None,
) -> torch.Tensor:
    resized = _resize_to_patch_grid(image.convert("RGB"), max_width)
    tensor = _image_to_dino_tensor(resized, device)
    with torch.no_grad():
        tokens = model.forward_features(tensor)["x_norm_patchtokens"][0]
        tokens = F.normalize(tokens.to(torch.float32), dim=-1)
    grid_h = resized.height // PATCH_SIZE
    grid_w = resized.width // PATCH_SIZE
    return tokens.reshape(grid_h, grid_w, -1)


def _extract_dino_tokens_from_array(
    rgb: np.ndarray,
    model,
    device: str,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    target_h = grid_h * PATCH_SIZE
    target_w = grid_w * PATCH_SIZE
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device=device, dtype=torch.float32).unsqueeze(0)
    if tuple(tensor.shape[-2:]) != (target_h, target_w):
        tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    tensor = (tensor.clamp(0.0, 1.0) - mean) / std
    with torch.no_grad():
        tokens = model.forward_features(tensor)["x_norm_patchtokens"][0]
        tokens = F.normalize(tokens.to(torch.float32), dim=-1)
    return tokens.reshape(grid_h, grid_w, -1)


def _smooth_2d(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return value
    if kernel_size % 2 == 0:
        raise ValueError("--smooth-kernel must be odd when greater than 1")
    value4 = value.reshape(1, 1, *value.shape[-2:])
    value4 = F.pad(value4, (kernel_size // 2,) * 4, mode="replicate")
    return F.avg_pool2d(value4, kernel_size=kernel_size, stride=1).view(*value.shape[-2:])


def _resize_map(value: np.ndarray, size_hw: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(value.astype(np.float32)).reshape(1, 1, *value.shape[-2:])
    resized = F.interpolate(tensor, size=size_hw, mode=mode, align_corners=False if mode in ("bilinear", "bicubic") else None)
    return resized.reshape(*size_hw).numpy()


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
    method_dir = _infer_method_dir(baseline_dir, args.split, args.method)
    render_to_image = _load_render_index(baseline_dir)

    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_manifest = None
    model = None
    cache_dir = Path(args.gt_cache) if args.gt_cache else None
    if cache_dir:
        cache_manifest = read_manifest(cache_dir)
        backend = cache_manifest.get("backend")
        max_width = cache_manifest.get("max_width")
    else:
        backend = args.backend
        max_width = args.max_width
    if not backend:
        raise ValueError("Unable to infer DINO backend. Pass --backend or --gt-cache.")

    device = args.device
    model = _load_dinov2_model(backend, args.dinov2_repo, device, pretrained=True)

    rows = []
    topk = float(args.topk)
    rgb_topk = float(args.rgb_topk)
    broad_topk = float(args.rgb_broad_topk)
    save_maps = bool(args.save_maps)
    maps_dir = output_dir / "maps"
    if save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    for render_path in sorted(render_dir.iterdir()):
        if render_path.suffix not in IMAGE_SUFFIXES:
            continue
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            raise FileNotFoundError("Missing GT image {}".format(gt_path))
        if render_path.name not in render_to_image:
            raise KeyError("Missing render index {} in cameras.json".format(render_path.name))

        image_name = render_to_image[render_path.name]
        render = _load_rgb(render_path)
        gt = _load_rgb(gt_path)
        rgb_error = np.mean(np.abs(render - gt), axis=-1).astype(np.float32)

        if cache_dir:
            gt_tokens_np, manifest, entry = _load_cache_tokens(cache_dir, image_name)
            gt_tokens = torch.from_numpy(gt_tokens_np).to(device=device, dtype=torch.float32)
            grid_size = tuple(gt_tokens.shape[:2])
            rendered_tokens = _extract_dino_tokens_from_array(render, model, device, grid_size)
            source_shape = entry.get("source_shape")
            cache_file = entry.get("cache_file")
        else:
            with Image.open(gt_path) as gt_image:
                gt_tokens = _extract_dino_tokens_from_image(gt_image, model, device, max_width)
            grid_size = tuple(gt_tokens.shape[:2])
            rendered_tokens = _extract_dino_tokens_from_array(render, model, device, grid_size)
            source_shape = list(gt.shape[:2])
            cache_file = ""

        gt_tokens = F.normalize(gt_tokens.to(torch.float32), dim=-1)
        patch_error = 0.5 * torch.clamp(
            1.0 - F.cosine_similarity(rendered_tokens, gt_tokens, dim=-1),
            min=0.0,
            max=2.0,
        )
        patch_error = _smooth_2d(patch_error, int(args.smooth_kernel))
        patch_error_np = patch_error.detach().cpu().numpy().astype(np.float32)
        dino_pixel = _resize_map(patch_error_np, rgb_error.shape, args.upsample)

        dino_norm = _normalize01(dino_pixel)
        rgb_norm = _normalize01(rgb_error)
        dino_mask = _topk_mask(dino_norm, topk)
        rgb_mask = _topk_mask(rgb_norm, rgb_topk)
        rgb_broad_mask = _topk_mask(rgb_norm, broad_topk)
        random_iou = _safe_ratio(topk * rgb_topk, topk + rgb_topk - topk * rgb_topk)

        overlap = dino_mask & rgb_mask
        union = dino_mask | rgb_mask
        broad_overlap = dino_mask & rgb_broad_mask
        row = {
            "view_index": int(render_path.stem),
            "render_name": render_path.name,
            "image_name": image_name,
            "grid_h": int(grid_size[0]),
            "grid_w": int(grid_size[1]),
            "cache_file": cache_file,
            "source_shape": json.dumps(source_shape),
            "rgb_l1": float(np.mean(rgb_error)),
            "dino_mean": float(np.mean(dino_pixel)),
            "dino_rgb_topk_iou": _safe_ratio(float(np.count_nonzero(overlap)), float(np.count_nonzero(union))),
            "dino_rgb_topk_recall": _safe_ratio(float(np.count_nonzero(overlap)), float(np.count_nonzero(rgb_mask))),
            "dino_rgb_topk_precision": _safe_ratio(float(np.count_nonzero(overlap)), float(np.count_nonzero(dino_mask))),
            "dino_rgb_random_iou": random_iou,
            "dino_rgb_iou_lift": _safe_ratio(
                _safe_ratio(float(np.count_nonzero(overlap)), float(np.count_nonzero(union))),
                random_iou,
            ),
            "dino_in_rgb_broad_recall": _safe_ratio(
                float(np.count_nonzero(broad_overlap)),
                float(np.count_nonzero(dino_mask)),
            ),
            "rgb_in_dino_l1": _masked_mean(rgb_error, dino_mask),
            "rgb_not_dino_l1": _masked_mean(rgb_error, ~dino_mask),
            "dino_in_rgb_topk_mean": _masked_mean(dino_norm, rgb_mask),
            "dino_not_rgb_topk_mean": _masked_mean(dino_norm, ~rgb_mask),
            "spearman_pixel": _spearman(dino_norm, rgb_norm),
            "spearman_patch": _spearman(
                patch_error_np,
                F.adaptive_avg_pool2d(
                    torch.from_numpy(rgb_error).reshape(1, 1, *rgb_error.shape),
                    output_size=grid_size,
                )
                .reshape(*grid_size)
                .numpy(),
            ),
        }
        rows.append(row)

        if save_maps:
            stem = render_path.stem
            np.save(maps_dir / "{}_dino_pixel.npy".format(stem), dino_pixel.astype(np.float32))
            np.save(maps_dir / "{}_rgb_error.npy".format(stem), rgb_error.astype(np.float32))
            Image.fromarray(np.clip(dino_norm * 255.0, 0, 255).astype(np.uint8)).save(
                maps_dir / "{}_dino_pixel.png".format(stem)
            )
            Image.fromarray(np.clip(rgb_norm * 255.0, 0, 255).astype(np.uint8)).save(
                maps_dir / "{}_rgb_error.png".format(stem)
            )

    if not rows:
        raise ValueError("No render images found in {}".format(render_dir))

    summary = _summarize(rows)
    summary.update(
        {
            "baseline_model": str(baseline_dir),
            "split": args.split,
            "method": method_dir.name,
            "gt_cache": str(cache_dir) if cache_dir else None,
            "backend": backend,
            "max_width": max_width,
            "dinov2_repo": args.dinov2_repo,
            "device": device,
            "topk": topk,
            "rgb_topk": rgb_topk,
            "rgb_broad_topk": broad_topk,
            "smooth_kernel": int(args.smooth_kernel),
            "upsample": args.upsample,
        }
    )
    _write_csv(output_dir / "per_view.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose train-style render-vs-GT DINO descriptor residual overlap with RGB error."
    )
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default=None)
    parser.add_argument("--gt-cache", default=None, help="Existing GT/source DINO patch-token cache.")
    parser.add_argument("--backend", default="dinov2_vits14", help="DINO backend when --gt-cache is not provided.")
    parser.add_argument("--max-width", type=int, default=None, help="DINO max width when --gt-cache is not provided.")
    parser.add_argument("--dinov2-repo", default="output/0001/external/dinov2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=float, default=0.25)
    parser.add_argument("--rgb-topk", type=float, default=0.25)
    parser.add_argument("--rgb-broad-topk", type=float, default=0.50)
    parser.add_argument("--smooth-kernel", type=int, default=1)
    parser.add_argument("--upsample", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = diagnose(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
