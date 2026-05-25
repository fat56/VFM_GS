#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_0001_fastgs_big_eval import (
    point_count_at_iteration,
    render_iteration,
    require_success,
    run_command,
    run_metrics,
    scene_overrides,
)


DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
DEFAULT_CONFIG = "configs/experiments/0019_descriptor_or_rgb_clone_baseline30_full9.yaml"
DEFAULT_CACHE_TEMPLATE = "output/0001/vfm_cache/{scene}_dinov2_vits14"
DEFAULT_BASELINE_CSV = "output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv"
DEFAULT_RUN_NAME = "descriptor_or_rgb_clone0k15k_baseline30_r_auto"
DEFAULT_METHOD = "descriptor_or_rgb_clone0k15k_baseline30"
EVAL_ITERATIONS = [15000, 20000, 25000, 30000]
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")


def read_metrics(run_dir: Path, iteration: int) -> dict[str, float | None]:
    path = run_dir / "results.json"
    if not path.exists():
        return {key: None for key in METRIC_KEYS}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("ours_{}".format(iteration), {key: None for key in METRIC_KEYS})


def metrics_complete(run_dir: Path, iterations: list[int]) -> bool:
    for iteration in iterations:
        metrics = read_metrics(run_dir, iteration)
        if not all(metrics[key] is not None for key in METRIC_KEYS):
            return False
    return True


def parse_train_log(log_path: Path) -> tuple[int | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gs_matches = re.findall(r"Gaussian number:\s*(\d+)", text)
    time_matches = re.findall(r"Training time:\s*([0-9.]+)", text)
    gs_num = int(gs_matches[-1]) if gs_matches else None
    train_time = float(time_matches[-1]) if time_matches else None
    return gs_num, train_time


def stage_name(iteration: int) -> str:
    if iteration == 15000:
        return "densify_end_15k"
    if iteration == 20000:
        return "final_prune_tail_20k"
    if iteration == 25000:
        return "final_prune_tail_25k"
    if iteration == 30000:
        return "final_30k"
    return "iter_{}".format(iteration)


def train_scene(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    cache_dir: Path,
) -> None:
    save_iterations = EVAL_ITERATIONS
    if all(point_count_at_iteration(run_dir, iteration) is not None for iteration in save_iterations):
        return

    cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.train",
        "--variant",
        args.variant,
        "--config",
        args.config,
        "-s",
        str(scene_path),
        "-i",
        args.train_images,
        "-m",
        str(run_dir),
        "--eval",
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--checkpoint_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--densification_interval",
        str(args.densification_interval),
        "--optimizer_type",
        "default",
        "-r",
        str(args.resolution),
        "--vfm_cache_dir",
        str(cache_dir),
    ]
    cmd.extend(overrides)
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def render_all_then_metrics(run_dir: Path, log_dir: Path, repo: Path, overrides: list[str]) -> None:
    if metrics_complete(run_dir, EVAL_ITERATIONS):
        return
    for iteration in EVAL_ITERATIONS:
        metrics = read_metrics(run_dir, iteration)
        if all(metrics[key] is not None for key in METRIC_KEYS):
            continue
        render_iteration(run_dir, log_dir, repo, overrides, iteration)
    run_metrics(run_dir, log_dir, repo)
    if not metrics_complete(run_dir, EVAL_ITERATIONS):
        missing = [
            iteration
            for iteration in EVAL_ITERATIONS
            if not all(read_metrics(run_dir, iteration)[key] is not None for key in METRIC_KEYS)
        ]
        raise RuntimeError("Metrics incomplete for {}: {}".format(run_dir, missing))


def row_for(
    dataset: str,
    scene: str,
    args: argparse.Namespace,
    iteration: int,
    run_dir: Path,
    train_time: float | None,
    cache_dir: Path,
) -> dict[str, object]:
    metrics = read_metrics(run_dir, iteration)
    return {
        "dataset": dataset,
        "scene": scene,
        "variant": args.variant_key,
        "method": args.method_name,
        "backend": "dinov2_descriptor_cosine",
        "branch": "clone_rgb_or_descriptor_0_15k_split_prune_fastgs_30k",
        "iteration": iteration,
        "relative_to_15k": iteration - 15000,
        "stage": stage_name(iteration),
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": point_count_at_iteration(run_dir, iteration),
        "train_time_s": train_time if iteration == args.iterations else None,
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_start_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = []
    by_scene: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene"]), {})[int(row["iteration"])] = row

    for scene, scene_rows in sorted(by_scene.items()):
        start = scene_rows.get(15000)
        if start is None:
            continue
        for iteration in (20000, 25000, 30000):
            target = scene_rows.get(iteration)
            if target is None:
                continue
            comparisons.append(delta_row(scene, start, target, start_iteration=15000))
    return comparisons


def delta_row(scene: str, start: dict[str, object], target: dict[str, object], start_iteration: int) -> dict[str, object]:
    return {
        "variant": target["variant"],
        "method": target["method"],
        "scene": scene,
        "start_iteration": start_iteration,
        "iteration": target["iteration"],
        "relative_iteration": int(target["iteration"]) - start_iteration,
        "stage": target["stage"],
        "delta_psnr": float(target["psnr"]) - float(start["psnr"]),
        "delta_ssim": float(target["ssim"]) - float(start["ssim"]),
        "delta_lpips": float(target["lpips"]) - float(start["lpips"]),
        "delta_gs_num": int(target["gs_num"]) - int(start["gs_num"]),
        "start_psnr": start["psnr"],
        "target_psnr": target["psnr"],
        "start_ssim": start["ssim"],
        "target_ssim": target["ssim"],
        "start_lpips": start["lpips"],
        "target_lpips": target["lpips"],
        "start_gs_num": start["gs_num"],
        "target_gs_num": target["gs_num"],
        "run_dir": target["run_dir"],
    }


def build_delta_aggregate(comparisons: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = []
    for relative_iteration in sorted({int(row["relative_iteration"]) for row in comparisons}):
        subset = [row for row in comparisons if int(row["relative_iteration"]) == relative_iteration]
        if not subset:
            continue
        row = {
            "variant": subset[0]["variant"],
            "method": subset[0]["method"],
            "relative_iteration": relative_iteration,
            "scene_count": len(subset),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            values = [float(item[key]) for item in subset]
            row["avg_" + key] = sum(values) / len(values)
        aggregate.append(row)
    return aggregate


def read_baseline_curve(path: Path, scenes: set[str]) -> dict[tuple[str, int], dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "mipnerf360":
                continue
            if row.get("scene") not in scenes:
                continue
            iteration = int(row.get("iteration", "0"))
            rows[(str(row["scene"]), iteration)] = row
    return rows


def build_baseline_comparison(
    rows: list[dict[str, object]],
    baseline_csv: Path,
    iterations: set[int] | None = None,
) -> list[dict[str, object]]:
    scenes = {str(row["scene"]) for row in rows}
    baseline = read_baseline_curve(baseline_csv, scenes)
    out = []
    for row in rows:
        iteration = int(row["iteration"])
        if iterations is not None and iteration not in iterations:
            continue
        base = baseline.get((str(row["scene"]), iteration))
        if base is None:
            continue
        out.append(
            {
                "variant": row["variant"],
                "method": row["method"],
                "scene": row["scene"],
                "iteration": row["iteration"],
                "stage": row["stage"],
                "baseline_method": base["method"],
                "delta_psnr": float(row["psnr"]) - float(base["psnr"]),
                "delta_ssim": float(row["ssim"]) - float(base["ssim"]),
                "delta_lpips": float(row["lpips"]) - float(base["lpips"]),
                "delta_gs_num": int(row["gs_num"]) - int(base["gs_num"]),
                "baseline_psnr": base["psnr"],
                "target_psnr": row["psnr"],
                "baseline_ssim": base["ssim"],
                "target_ssim": row["ssim"],
                "baseline_lpips": base["lpips"],
                "target_lpips": row["lpips"],
                "baseline_gs_num": base["gs_num"],
                "target_gs_num": row["gs_num"],
                "run_dir": row["run_dir"],
            }
        )
    return out


def build_iteration_aggregate(comparisons: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = []
    for iteration in sorted({int(row["iteration"]) for row in comparisons}):
        subset = [row for row in comparisons if int(row["iteration"]) == iteration]
        if not subset:
            continue
        row = {
            "variant": subset[0]["variant"],
            "method": subset[0]["method"],
            "iteration": iteration,
            "scene_count": len(subset),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            values = [float(item[key]) for item in subset]
            row["avg_" + key] = sum(values) / len(values)
        aggregate.append(row)
    return aggregate


def write_summaries(output_root: Path, rows: list[dict[str, object]], baseline_csv: Path) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    start_comparisons = build_start_comparisons(rows)
    write_csv(output_root / "comparison_15k_to_eval.csv", start_comparisons)
    (output_root / "comparison_15k_to_eval.json").write_text(
        json.dumps(start_comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    start_aggregate = build_delta_aggregate(start_comparisons)
    write_csv(output_root / "aggregate_15k_to_eval.csv", start_aggregate)
    (output_root / "aggregate_15k_to_eval.json").write_text(
        json.dumps(start_aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_curve = build_baseline_comparison(rows, baseline_csv)
    write_csv(output_root / "comparison_vs_baseline_curve.csv", baseline_curve)
    (output_root / "comparison_vs_baseline_curve.json").write_text(
        json.dumps(baseline_curve, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline_curve_aggregate = build_iteration_aggregate(baseline_curve)
    write_csv(output_root / "aggregate_vs_baseline_curve.csv", baseline_curve_aggregate)
    (output_root / "aggregate_vs_baseline_curve.json").write_text(
        json.dumps(baseline_curve_aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline30 = build_baseline_comparison(rows, baseline_csv, iterations={30000})
    write_csv(output_root / "comparison_vs_baseline30.csv", baseline30)
    (output_root / "comparison_vs_baseline30.json").write_text(
        json.dumps(baseline30, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline30_aggregate = build_iteration_aggregate(baseline30)
    write_csv(output_root / "aggregate_vs_baseline30.csv", baseline30_aggregate)
    (output_root / "aggregate_vs_baseline30.json").write_text(
        json.dumps(baseline30_aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0019 descriptor OR RGB clone full9 baseline-30K experiment.")
    parser.add_argument("--dataset-name", default="mipnerf360", choices=["mipnerf360"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--cache-template", default=DEFAULT_CACHE_TEMPLATE)
    parser.add_argument("--baseline-csv", type=Path, default=Path(DEFAULT_BASELINE_CSV))
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variant-key", default="0019_descriptor_or_rgb_clone_baseline30_full9")
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--no-scene-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    use_overrides = not args.no_scene_overrides

    for scene in args.scenes:
        scene_path = args.dataset_root / scene
        if not scene_path.exists():
            raise FileNotFoundError(scene_path)
        overrides = scene_overrides(args.dataset_name, scene, use_overrides)
        cache_dir = Path(args.cache_template.format(scene=scene))
        if not (cache_dir / "manifest.json").exists():
            raise FileNotFoundError(cache_dir / "manifest.json")

        run_dir = args.output_root / scene / args.run_name
        log_dir = args.output_root / scene / "logs" / args.run_name
        print(
            "[0019] {} descriptor-or-rgb clone eval {}".format(
                scene,
                ",".join(str(iteration) for iteration in EVAL_ITERATIONS),
            ),
            flush=True,
        )

        train_scene(scene_path, run_dir, log_dir, repo, args, overrides, cache_dir)
        render_all_then_metrics(run_dir, log_dir, repo, overrides)
        _, train_time = parse_train_log(log_dir / "train.log")

        for iteration in EVAL_ITERATIONS:
            rows.append(row_for(args.dataset_name, scene, args, iteration, run_dir, train_time, cache_dir))
        write_summaries(args.output_root, rows, args.baseline_csv)

    write_summaries(args.output_root, rows, args.baseline_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
