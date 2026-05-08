#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips")
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
        raise RuntimeError("Train selector evaluation requires FastGS runtime dependencies.") from RUNTIME_IMPORT_ERROR


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


def quality_gain(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    d_psnr = delta(optional_float(candidate["psnr"]), optional_float(reference["psnr"]))
    d_ssim = delta(optional_float(candidate["ssim"]), optional_float(reference["ssim"]))
    d_lpips = delta(optional_float(candidate["lpips"]), optional_float(reference["lpips"]))
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


def qcgi(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    gain = quality_gain(candidate, reference)
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


def evaluate_train_split(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    require_runtime_imports()
    run_dir = Path(row["run_dir"])
    cfg = load_cfg_args(run_dir)
    gaussians = GaussianModel(cfg.sh_degree, optimizer_type=getattr(cfg, "optimizer_type", "default"))
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = select_views(scene.getTrainCameras(), args.max_views, args.view_stride)

    psnrs = []
    ssims = []
    lpipss = []
    with torch.no_grad():
        for view in tqdm(views, desc="{} {} train".format(row["scene"], row["method"])):
            rendering = render_fastgs(view, gaussians, cfg, background, getattr(cfg, "mult", 0.5))["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0).unsqueeze(0)
            gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0).unsqueeze(0).cuda()
            ssims.append(float(ssim(rendering, gt)))
            psnrs.append(float(psnr(rendering, gt)))
            lpipss.append(float(lpips(rendering, gt, net_type="vgg")))

    return {
        "dataset": row["dataset"],
        "scene": row["scene"],
        "method": row["method"],
        "split": "train",
        "view_count": len(views),
        "psnr": mean(psnrs),
        "ssim": mean(ssims),
        "lpips": mean(lpipss),
        "gs_num": optional_int(row.get("gs_num")),
        "train_time_s": optional_float(row.get("train_time_s")),
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


def psnr_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        optional_float(row["psnr"]) or float("-inf"),
        -(optional_float(row["lpips"]) or float("inf")),
        optional_float(row["ssim"]) or float("-inf"),
    )


def qcgi_key(row: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[float, float, float]:
    if baseline is None or row["method"] == "baseline":
        score = 0.0
    else:
        score = qcgi(row, baseline)
        if score is None:
            score = float("-inf")
    return (score, optional_float(row["psnr"]) or float("-inf"), -(optional_float(row["lpips"]) or float("inf")))


def lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["scene"]), str(row["method"])): row
        for row in rows
    }


def build_train_selector_rows(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        by_scene[(str(row["dataset"]), str(row["scene"]))][str(row["method"])] = row
    test_by_key = lookup(test_rows)

    selector_rows = []
    for (dataset, scene), methods in sorted(by_scene.items()):
        ordered = list(methods.values())
        baseline = methods.get("baseline")
        train_best_psnr = max(ordered, key=psnr_key)
        train_qcgi = max(ordered, key=lambda row: qcgi_key(row, baseline))
        for selector_name, picked in (
            ("train_best_psnr", train_best_psnr),
            ("train_qcgi", train_qcgi),
        ):
            test_row = test_by_key.get((dataset, scene, str(picked["method"])))
            if test_row is None:
                continue
            selector_rows.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "selector": selector_name,
                    "picked_method": picked["method"],
                    "train_psnr": picked["psnr"],
                    "train_ssim": picked["ssim"],
                    "train_lpips": picked["lpips"],
                    "train_qcgi": 0.0 if picked["method"] == "baseline" or baseline is None else qcgi(picked, baseline),
                    "test_psnr": optional_float(test_row["psnr"]),
                    "test_ssim": optional_float(test_row["ssim"]),
                    "test_lpips": optional_float(test_row["lpips"]),
                    "test_gs_num": optional_int(test_row["gs_num"]),
                    "test_train_time_s": optional_float(test_row["train_time_s"]),
                    "run_dir": test_row["run_dir"],
                }
            )
    return selector_rows


def build_selector_averages(selector_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        grouped[str(row["selector"])].append(row)

    averages = []
    for selector, rows in sorted(grouped.items()):
        avg: dict[str, Any] = {"selector": selector, "scene_count": len(rows)}
        for key in ("test_psnr", "test_ssim", "test_lpips", "test_gs_num", "test_train_time_s"):
            values = [float(row[key]) for row in rows if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = mean(values)
        averages.append(avg)
    return averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-split selectors for experiment 0001.")
    parser.add_argument("--input-summary", type=Path, default=Path("output/0001/cross_dataset_selector/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/0001/train_selector"))
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--view-stride", type=int, default=1)
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
    train_metrics_path = args.output_dir / "train_metrics.csv"
    train_rows: list[dict[str, Any]] = []
    if args.resume and train_metrics_path.exists():
        train_rows.extend(read_csv(train_metrics_path))
    seen = existing_keys(train_metrics_path) if args.resume else set()

    for row in input_rows:
        key = (row["dataset"], row["scene"], row["method"])
        if key in seen:
            continue
        print("[{}:{}:{}] evaluating train split".format(row["dataset"], row["scene"], row["method"]), flush=True)
        train_rows.append(evaluate_train_split(row, args))
        write_csv(train_metrics_path, train_rows)
        write_json(args.output_dir / "train_metrics.json", train_rows)

    test_rows = [dict(row) for row in read_csv(args.input_summary)]
    selector_rows = build_train_selector_rows(train_rows, test_rows)
    selector_averages = build_selector_averages(selector_rows)
    write_csv(args.output_dir / "train_selector_recommendations.csv", selector_rows)
    write_json(args.output_dir / "train_selector_recommendations.json", selector_rows)
    write_csv(args.output_dir / "train_selector_averages.csv", selector_averages)
    write_json(args.output_dir / "train_selector_averages.json", selector_averages)

    print("Wrote {}".format(train_metrics_path))
    print("Wrote {}".format(args.output_dir / "train_selector_averages.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
