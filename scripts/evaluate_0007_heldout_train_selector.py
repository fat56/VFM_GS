#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any


QUALITY_WEIGHTS = {
    "psnr": 1.0,
    "ssim": 20.0,
    "lpips": 5.0,
}
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04


RUNTIME_IMPORT_ERROR = None

try:
    import torch
    from tqdm import tqdm

    from vfm_gs.gaussian_renderer import GaussianModel
    from vfm_gs.gaussian_renderer import render_fastgs
    from vfm_gs.lpips_pytorch import lpips
    from vfm_gs.scene import Scene
    from vfm_gs.utils.image_utils import psnr
    from vfm_gs.utils.loss_utils import ssim
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc


def require_runtime_imports() -> None:
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError("Held-out selector evaluation requires FastGS runtime dependencies.") from RUNTIME_IMPORT_ERROR


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


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def delta(value: float | int | None, reference: float | int | None) -> float | int | None:
    if value is None or reference is None:
        return None
    return value - reference


def quality_gain(candidate: dict[str, Any], reference: dict[str, Any], prefix: str) -> float | None:
    d_psnr = delta(optional_float(candidate[f"{prefix}_psnr"]), optional_float(reference[f"{prefix}_psnr"]))
    d_ssim = delta(optional_float(candidate[f"{prefix}_ssim"]), optional_float(reference[f"{prefix}_ssim"]))
    d_lpips = delta(optional_float(candidate[f"{prefix}_lpips"]), optional_float(reference[f"{prefix}_lpips"]))
    if d_psnr is None or d_ssim is None or d_lpips is None:
        return None
    return (
        QUALITY_WEIGHTS["psnr"] * d_psnr
        + QUALITY_WEIGHTS["ssim"] * d_ssim
        - QUALITY_WEIGHTS["lpips"] * d_lpips
    )


def gs_penalty(delta_gs_num: int | float | None) -> float | None:
    if delta_gs_num is None:
        return None
    growth = max(0.0, float(delta_gs_num))
    soft_growth = min(growth, GS_SOFT_BUDGET)
    heavy_growth = max(0.0, growth - GS_SOFT_BUDGET)
    return (
        GS_PENALTY_PER_10K * (soft_growth / GS_UNIT)
        + GS_HEAVY_PENALTY_PER_10K * (heavy_growth / GS_UNIT)
    )


def qcgi(candidate: dict[str, Any], reference: dict[str, Any], prefix: str) -> float | None:
    gain = quality_gain(candidate, reference, prefix)
    penalty = gs_penalty(delta(optional_int(candidate["gs_num"]), optional_int(reference["gs_num"])))
    if gain is None or penalty is None:
        return None
    return gain - penalty


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


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


def evaluate_candidate(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    require_runtime_imports()
    run_dir = Path(row["run_dir"])
    cfg = load_cfg_args(run_dir)
    gaussians = GaussianModel(cfg.sh_degree, optimizer_type=getattr(cfg, "optimizer_type", "default"))
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = select_views(scene.getTrainCameras(), args.max_views, args.view_stride)

    selector_metrics = {"psnr": [], "ssim": [], "lpips": []}
    holdout_metrics = {"psnr": [], "ssim": [], "lpips": []}

    with torch.no_grad():
        for view_index, view in enumerate(tqdm(views, desc="{} {} train".format(row["scene"], row["method"]))):
            rendering = render_fastgs(view, gaussians, cfg, background, getattr(cfg, "mult", 0.5))["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0).unsqueeze(0)
            gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0).unsqueeze(0).cuda()
            metric_row = {
                "psnr": float(psnr(rendering, gt)),
                "ssim": float(ssim(rendering, gt)),
                "lpips": float(lpips(rendering, gt, net_type="vgg")),
            }
            bucket = selector_metrics if view_index % 2 == 0 else holdout_metrics
            for key, value in metric_row.items():
                bucket[key].append(value)

    if not holdout_metrics["psnr"] and selector_metrics["psnr"]:
        holdout_metrics = {key: values[:] for key, values in selector_metrics.items()}

    return {
        "dataset": row["dataset"],
        "scene": row["scene"],
        "method": row["method"],
        "selector_view_count": len(selector_metrics["psnr"]),
        "selector_psnr": mean(selector_metrics["psnr"]),
        "selector_ssim": mean(selector_metrics["ssim"]),
        "selector_lpips": mean(selector_metrics["lpips"]),
        "holdout_view_count": len(holdout_metrics["psnr"]),
        "holdout_psnr": mean(holdout_metrics["psnr"]),
        "holdout_ssim": mean(holdout_metrics["ssim"]),
        "holdout_lpips": mean(holdout_metrics["lpips"]),
        "gs_num": optional_int(row.get("gs_num")),
        "train_time_s": optional_float(row.get("train_time_s")),
        "test_psnr": optional_float(row.get("psnr")),
        "test_ssim": optional_float(row.get("ssim")),
        "test_lpips": optional_float(row.get("lpips")),
        "test_gs_num": optional_int(row.get("gs_num")),
        "test_train_time_s": optional_float(row.get("train_time_s")),
        "run_dir": row["run_dir"],
    }


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    datasets = set(args.datasets or [])
    scenes = set(args.scenes or [])
    methods = set(args.methods or [])
    filtered = []
    for row in rows:
        if datasets and row["dataset"] not in datasets:
            continue
        if scenes and row["scene"] not in scenes:
            continue
        if methods and row["method"] not in methods:
            continue
        filtered.append(row)
    return filtered


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    return {
        (row["dataset"], row["scene"], row["method"])
        for row in read_csv(path)
    }


def lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["scene"]), str(row["method"])): row
        for row in rows
    }


def selector_psnr_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        optional_float(row["selector_psnr"]) or float("-inf"),
        -(optional_float(row["selector_lpips"]) or float("inf")),
        optional_float(row["selector_ssim"]) or float("-inf"),
    )


def selector_qcgi_key(row: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[float, float, float]:
    if baseline is None or row["method"] == "baseline":
        score = 0.0
    else:
        score = qcgi(row, baseline, "selector")
        if score is None:
            score = float("-inf")
    return (
        score,
        optional_float(row["selector_psnr"]) or float("-inf"),
        -(optional_float(row["selector_lpips"]) or float("inf")),
    )


def annotate_selector_qcgi(rows: list[dict[str, Any]]) -> None:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[(str(row["dataset"]), str(row["scene"]))][str(row["method"])] = row
    for methods in by_scene.values():
        baseline = methods.get("baseline")
        for row in methods.values():
            if row["method"] == "baseline" or baseline is None:
                row["selector_qcgi"] = 0.0
            else:
                row["selector_qcgi"] = qcgi(row, baseline, "selector")


def build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[(str(row["dataset"]), str(row["scene"]))][str(row["method"])] = row

    recommendations: list[dict[str, Any]] = []
    for (dataset, scene), methods in sorted(by_scene.items()):
        ordered = list(methods.values())
        baseline = methods.get("baseline")
        selector_best_psnr = max(ordered, key=selector_psnr_key)
        selector_qcgi_pick = max(ordered, key=lambda row: selector_qcgi_key(row, baseline))
        for selector_name, picked in (
            ("selector_best_psnr", selector_best_psnr),
            ("selector_qcgi", selector_qcgi_pick),
        ):
            baseline_row = baseline if baseline is not None else picked
            recommendations.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "selector": selector_name,
                    "picked_method": picked["method"],
                    "selector_view_count": picked["selector_view_count"],
                    "selector_psnr": picked["selector_psnr"],
                    "selector_ssim": picked["selector_ssim"],
                    "selector_lpips": picked["selector_lpips"],
                    "selector_qcgi": picked.get("selector_qcgi", 0.0),
                    "holdout_view_count": picked["holdout_view_count"],
                    "holdout_psnr": picked["holdout_psnr"],
                    "holdout_ssim": picked["holdout_ssim"],
                    "holdout_lpips": picked["holdout_lpips"],
                    "test_psnr": picked["test_psnr"],
                    "test_ssim": picked["test_ssim"],
                    "test_lpips": picked["test_lpips"],
                    "test_gs_num": picked["test_gs_num"],
                    "test_train_time_s": picked["test_train_time_s"],
                    "gs_num": picked["gs_num"],
                    "train_time_s": picked["train_time_s"],
                    "run_dir": picked["run_dir"],
                    "baseline_selector_psnr": baseline_row["selector_psnr"],
                    "baseline_selector_ssim": baseline_row["selector_ssim"],
                    "baseline_selector_lpips": baseline_row["selector_lpips"],
                    "baseline_holdout_psnr": baseline_row["holdout_psnr"],
                    "baseline_holdout_ssim": baseline_row["holdout_ssim"],
                    "baseline_holdout_lpips": baseline_row["holdout_lpips"],
                    "baseline_test_psnr": baseline_row["test_psnr"],
                    "baseline_test_ssim": baseline_row["test_ssim"],
                    "baseline_test_lpips": baseline_row["test_lpips"],
                    "baseline_gs_num": baseline_row["gs_num"],
                    "baseline_train_time_s": baseline_row["train_time_s"],
                }
            )
    return recommendations


def build_averages(rows: list[dict[str, Any]], selector_label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["selector"])].append(row)

    averages = []
    for selector, items in sorted(grouped.items()):
        avg: dict[str, Any] = {"selector": selector, "scene_count": len(items)}
        for key in (
            "selector_psnr",
            "selector_ssim",
            "selector_lpips",
            "selector_qcgi",
            "holdout_psnr",
            "holdout_ssim",
            "holdout_lpips",
            "test_psnr",
            "test_ssim",
            "test_lpips",
            "test_gs_num",
            "test_train_time_s",
            "gs_num",
            "train_time_s",
        ):
            values = [float(row[key]) for row in items if row[key] not in (None, "")]
            avg[f"avg_{key}"] = mean(values)
        avg["label"] = selector_label
        averages.append(avg)
    return averages


def build_baseline_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows = [row for row in rows if row["method"] == "baseline"]
    if not baseline_rows:
        return []
    avg: dict[str, Any] = {"selector": "baseline", "scene_count": len(baseline_rows)}
    for key in (
        "selector_psnr",
        "selector_ssim",
        "selector_lpips",
        "selector_qcgi",
        "holdout_psnr",
        "holdout_ssim",
        "holdout_lpips",
        "test_psnr",
        "test_ssim",
        "test_lpips",
        "test_gs_num",
        "test_train_time_s",
        "gs_num",
        "train_time_s",
    ):
        values = [float(row[key]) for row in baseline_rows if row[key] not in (None, "")]
        avg[f"avg_{key}"] = mean(values)
    return [avg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate held-out train-split selectors for experiment 0007.")
    parser.add_argument("--input-summary", type=Path, default=Path("output/0006/validation_selector/mipnerf360_depth_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/0007/heldout_selector"))
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--view-stride", type=int, default=7)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_runtime_imports()
    torch.cuda.set_device(torch.device("cuda:0"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_rows = filter_rows(read_csv(args.input_summary), args)
    metrics_path = args.output_dir / "heldout_candidate_metrics.csv"
    metrics_rows: list[dict[str, Any]] = []
    if args.resume and metrics_path.exists():
        metrics_rows.extend(read_csv(metrics_path))
    seen = existing_keys(metrics_path) if args.resume else set()

    for row in input_rows:
        key = (row["dataset"], row["scene"], row["method"])
        if key in seen:
            continue
        print("[{}:{}:{}] evaluating held-out train split".format(row["dataset"], row["scene"], row["method"]), flush=True)
        metrics_rows.append(evaluate_candidate(row, args))
        annotate_selector_qcgi(metrics_rows)
        write_csv(metrics_path, metrics_rows)
        write_json(args.output_dir / "heldout_candidate_metrics.json", metrics_rows)

    annotate_selector_qcgi(metrics_rows)
    recommendations = build_recommendations(metrics_rows)
    selector_averages = build_averages(recommendations, "selected")
    baseline_averages = build_baseline_averages(metrics_rows)

    write_csv(args.output_dir / "heldout_selector_recommendations.csv", recommendations)
    write_json(args.output_dir / "heldout_selector_recommendations.json", recommendations)
    write_csv(args.output_dir / "heldout_selector_averages.csv", selector_averages)
    write_json(args.output_dir / "heldout_selector_averages.json", selector_averages)
    write_csv(args.output_dir / "heldout_baseline_averages.csv", baseline_averages)
    write_json(args.output_dir / "heldout_baseline_averages.json", baseline_averages)

    print("Wrote {}".format(metrics_path))
    print("Wrote {}".format(args.output_dir / "heldout_selector_averages.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
