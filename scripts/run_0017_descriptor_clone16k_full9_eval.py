#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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


MIPNERF360_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
DEFAULT_BASELINE_TEMPLATE = (
    "output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/"
    "{scene}/fastgs_big_30k_curve_r_auto"
)
DEFAULT_CACHE_TEMPLATE = "output/0001/vfm_cache/{scene}_dinov2_vits14"
DEFAULT_CONFIG = "configs/experiments/0017_descriptor_clone16k_full9.yaml"
DEFAULT_VARIANT_KEY = "desc_clone_16k_full9"
DEFAULT_METHOD = "descriptor_clone_only_i16000_5k"
RUN_NAME_TEMPLATE = "{method}_i{start}_to_i{final}_5k_r_auto"
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


def copy_start_state(baseline_run_dir: Path, run_dir: Path, start_iteration: int) -> None:
    source_ply = baseline_run_dir / "point_cloud" / "iteration_{}".format(start_iteration) / "point_cloud.ply"
    if not source_ply.exists():
        raise FileNotFoundError("Missing baseline point cloud: {}".format(source_ply))

    target_dir = run_dir / "point_cloud" / "iteration_{}".format(start_iteration)
    target_ply = target_dir / "point_cloud.ply"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_ply.exists():
        shutil.copy2(source_ply, target_ply)

    for name in ("cfg_args", "cameras.json", "input.ply"):
        source = baseline_run_dir / name
        target = run_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def stage_name(relative_iteration: int, followup_iterations: int) -> str:
    if relative_iteration == 0:
        return "start"
    if relative_iteration == followup_iterations:
        return "end_5k"
    return "mid_{}k".format(relative_iteration // 1000)


def train_followup(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    eval_iterations: list[int],
    cache_dir: Path,
) -> None:
    final_iteration = eval_iterations[-1]
    save_iterations = eval_iterations[1:]
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
        str(final_iteration),
        "--start_pointcloud_iteration",
        str(args.start_iteration),
        "--test_iterations",
        str(final_iteration),
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--checkpoint_iterations",
        str(final_iteration),
        "--densification_interval",
        str(args.densification_interval),
        "--densify_from_iter",
        str(args.start_iteration),
        "--densify_until_iter",
        str(final_iteration),
        "--optimizer_type",
        "default",
        "-r",
        str(args.resolution),
        "--vfm_cache_dir",
        str(cache_dir),
    ]
    cmd.extend(overrides)
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def render_all_then_metrics(
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    overrides: list[str],
    eval_iterations: list[int],
) -> None:
    if metrics_complete(run_dir, eval_iterations):
        return
    for iteration in eval_iterations:
        metrics = read_metrics(run_dir, iteration)
        if all(metrics[key] is not None for key in METRIC_KEYS):
            continue
        render_iteration(run_dir, log_dir, repo, overrides, iteration)
    run_metrics(run_dir, log_dir, repo)
    if not metrics_complete(run_dir, eval_iterations):
        missing = [iteration for iteration in eval_iterations if not all(read_metrics(run_dir, iteration)[key] is not None for key in METRIC_KEYS)]
        raise RuntimeError("Metrics incomplete for {}: {}".format(run_dir, missing))


def row_for(
    dataset: str,
    scene: str,
    args: argparse.Namespace,
    stage: str,
    iteration: int,
    run_dir: Path,
    train_time: float | None,
    cache_dir: Path,
) -> dict[str, object]:
    metrics = read_metrics(run_dir, iteration)
    relative_iteration = iteration - args.start_iteration
    return {
        "dataset": dataset,
        "scene": scene,
        "variant": args.variant_key,
        "method": args.method_name,
        "backend": "descriptor",
        "branch": "clone_only",
        "start_iteration": args.start_iteration,
        "iteration": iteration,
        "relative_iteration": relative_iteration,
        "stage": stage,
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": point_count_at_iteration(run_dir, iteration),
        "train_time_s": train_time if stage == "end_5k" else None,
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


def build_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = []
    by_scene: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene"]), {})[int(row["relative_iteration"])] = row

    for scene, scene_rows in sorted(by_scene.items()):
        start = scene_rows.get(0)
        if start is None:
            continue
        for relative_iteration in sorted(value for value in scene_rows if value > 0):
            target = scene_rows[relative_iteration]
            comparisons.append(
                {
                    "variant": target["variant"],
                    "method": target["method"],
                    "backend": target["backend"],
                    "branch": target["branch"],
                    "scene": scene,
                    "start_iteration": target["start_iteration"],
                    "iteration": target["iteration"],
                    "relative_iteration": relative_iteration,
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
            )
    return comparisons


def build_aggregate(comparisons: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = []
    for relative_iteration in sorted({int(row["relative_iteration"]) for row in comparisons}):
        subset = [row for row in comparisons if int(row["relative_iteration"]) == relative_iteration]
        if not subset:
            continue
        row = {
            "variant": subset[0]["variant"],
            "method": subset[0]["method"],
            "backend": subset[0]["backend"],
            "branch": subset[0]["branch"],
            "relative_iteration": relative_iteration,
            "scene_count": len(subset),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            values = [float(item[key]) for item in subset]
            row["avg_" + key] = sum(values) / len(values)
        aggregate.append(row)
    return aggregate


def write_summaries(output_root: Path, rows: list[dict[str, object]]) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    comparisons = build_comparisons(rows)
    write_csv(output_root / "comparison_start_to_eval.csv", comparisons)
    (output_root / "comparison_start_to_eval.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    aggregate = build_aggregate(comparisons)
    write_csv(output_root / "aggregate_by_iteration.csv", aggregate)
    (output_root / "aggregate_by_iteration.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0017 descriptor clone-only 16K to 21K full9 continuation.")
    parser.add_argument("--dataset-name", default="mipnerf360", choices=["mipnerf360"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=MIPNERF360_SCENES)
    parser.add_argument("--baseline-template", default=DEFAULT_BASELINE_TEMPLATE)
    parser.add_argument("--cache-template", default=DEFAULT_CACHE_TEMPLATE)
    parser.add_argument("--start-iteration", type=int, default=16000)
    parser.add_argument("--followup-iterations", type=int, default=5000)
    parser.add_argument("--eval-step", type=int, default=1000)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--variant-key", default=DEFAULT_VARIANT_KEY)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--no-scene-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    use_overrides = not args.no_scene_overrides
    eval_offsets = list(range(0, int(args.followup_iterations) + 1, int(args.eval_step)))
    if eval_offsets[-1] != int(args.followup_iterations):
        eval_offsets.append(int(args.followup_iterations))
    eval_iterations = [int(args.start_iteration) + offset for offset in eval_offsets]
    final_iteration = eval_iterations[-1]

    for scene in args.scenes:
        scene_path = args.dataset_root / scene
        if not scene_path.exists():
            raise FileNotFoundError(scene_path)
        overrides = scene_overrides(args.dataset_name, scene, use_overrides)
        baseline_run_dir = Path(args.baseline_template.format(scene=scene))
        if not baseline_run_dir.exists():
            raise FileNotFoundError(baseline_run_dir)
        cache_dir = Path(args.cache_template.format(scene=scene))
        if not (cache_dir / "manifest.json").exists():
            raise FileNotFoundError(cache_dir / "manifest.json")

        run_name = RUN_NAME_TEMPLATE.format(
            method=args.method_name,
            start=args.start_iteration,
            final=final_iteration,
        )
        run_dir = args.output_root / scene / run_name
        log_dir = args.output_root / scene / "logs" / run_name
        print(
            "[0017] {} i{} -> i{} eval {}".format(
                scene,
                args.start_iteration,
                final_iteration,
                ",".join(str(iteration) for iteration in eval_iterations),
            ),
            flush=True,
        )

        copy_start_state(baseline_run_dir, run_dir, args.start_iteration)
        train_followup(scene_path, run_dir, log_dir, repo, args, overrides, eval_iterations, cache_dir)
        render_all_then_metrics(run_dir, log_dir, repo, overrides, eval_iterations)
        _, train_time = parse_train_log(log_dir / "train.log")

        for iteration in eval_iterations:
            relative_iteration = iteration - args.start_iteration
            rows.append(
                row_for(
                    args.dataset_name,
                    scene,
                    args,
                    stage_name(relative_iteration, args.followup_iterations),
                    iteration,
                    run_dir,
                    train_time,
                    cache_dir,
                )
            )
        write_summaries(args.output_root, rows)

    write_summaries(args.output_root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
