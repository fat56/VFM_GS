#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
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


DEFAULT_SCENES = ["room", "counter", "kitchen", "bonsai"]
DEFAULT_BASELINE30_CSV = "output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv"
DEFAULT_0017_TEMPLATE = (
    "output/0017/descriptor_clone16k_full9/mip_g1/{scene}/"
    "descriptor_clone_only_i16000_5k_i16000_to_i21000_5k_r_auto"
)
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")


@dataclass(frozen=True)
class ControlVariant:
    key: str
    method: str
    mode: str
    config: str
    run_name: str
    start_iteration: int
    eval_iterations: tuple[int, ...]
    source_template: str | None = None


VARIANTS: dict[str, ControlVariant] = {
    "rgb_fastgs_extra_fulltrain": ControlVariant(
        key="rgb_fastgs_extra_fulltrain",
        method="rgb_fastgs_extra15k20k_prune35k_fulltrain",
        mode="fulltrain",
        config="configs/experiments/0018_rgb_fastgs_extra_densify_fulltrain_prune35k_indoor.yaml",
        run_name="rgb_fastgs_extra15k20k_prune35k_fulltrain_r_auto",
        start_iteration=20000,
        eval_iterations=(15000, 20000, 25000, 30000, 35000),
    ),
    "desc16k21k_prune35k": ControlVariant(
        key="desc16k21k_prune35k",
        method="desc16k21k_from0017_prune35k",
        mode="continuation",
        config="configs/experiments/0018_desc16k21k_from0017_prune35k_indoor.yaml",
        run_name="desc16k21k_from0017_prune35k_r_auto",
        start_iteration=21000,
        eval_iterations=(21000, 25000, 30000, 35000),
        source_template=DEFAULT_0017_TEMPLATE,
    ),
}


def read_metrics(run_dir: Path, iteration: int) -> dict[str, float | None]:
    path = run_dir / "results.json"
    if not path.exists():
        return {key: None for key in METRIC_KEYS}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("ours_{}".format(iteration), {key: None for key in METRIC_KEYS})


def metrics_complete(run_dir: Path, iterations: tuple[int, ...]) -> bool:
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


def copy_start_state(source_run_dir: Path, run_dir: Path, iteration: int) -> None:
    source_ply = source_run_dir / "point_cloud" / "iteration_{}".format(iteration) / "point_cloud.ply"
    if not source_ply.exists():
        raise FileNotFoundError("Missing source point cloud: {}".format(source_ply))

    target_dir = run_dir / "point_cloud" / "iteration_{}".format(iteration)
    target_ply = target_dir / "point_cloud.ply"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_ply.exists():
        shutil.copy2(source_ply, target_ply)

    for name in ("cfg_args", "cameras.json", "input.ply"):
        source = source_run_dir / name
        target = run_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def stage_name(variant: ControlVariant, iteration: int) -> str:
    if variant.key == "rgb_fastgs_extra_fulltrain":
        if iteration == 15000:
            return "baseline_15k"
        if iteration == 20000:
            return "rgb_fastgs_extra_end_20k"
    if variant.key == "desc16k21k_prune35k" and iteration == 21000:
        return "source_0017_21k"
    return "prune_tail_{}k".format(iteration // 1000)


def train_fulltrain(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    variant: ControlVariant,
) -> None:
    if all(point_count_at_iteration(run_dir, iteration) is not None for iteration in variant.eval_iterations):
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
        variant.config,
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
        *[str(iteration) for iteration in variant.eval_iterations],
        "--checkpoint_iterations",
        *[str(iteration) for iteration in variant.eval_iterations],
        "--densification_interval",
        str(args.densification_interval),
        "--optimizer_type",
        "default",
        "-r",
        str(args.resolution),
    ]
    cmd.extend(overrides)
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def train_continuation(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    variant: ControlVariant,
    source_run_dir: Path,
) -> None:
    copy_start_state(source_run_dir, run_dir, variant.start_iteration)
    if all(point_count_at_iteration(run_dir, iteration) is not None for iteration in variant.eval_iterations):
        return
    save_iterations = [iteration for iteration in variant.eval_iterations if iteration > variant.start_iteration]
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
        variant.config,
        "-s",
        str(scene_path),
        "-i",
        args.train_images,
        "-m",
        str(run_dir),
        "--eval",
        "--iterations",
        str(args.iterations),
        "--start_pointcloud_iteration",
        str(variant.start_iteration),
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
    ]
    cmd.extend(overrides)
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def render_all_then_metrics(
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    overrides: list[str],
    eval_iterations: tuple[int, ...],
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
        missing = [
            iteration
            for iteration in eval_iterations
            if not all(read_metrics(run_dir, iteration)[key] is not None for key in METRIC_KEYS)
        ]
        raise RuntimeError("Metrics incomplete for {}: {}".format(run_dir, missing))


def row_for(
    dataset: str,
    scene: str,
    variant: ControlVariant,
    iteration: int,
    run_dir: Path,
    train_time: float | None,
    source_run_dir: Path | None,
) -> dict[str, object]:
    metrics = read_metrics(run_dir, iteration)
    return {
        "dataset": dataset,
        "scene": scene,
        "variant": variant.key,
        "method": variant.method,
        "mode": variant.mode,
        "start_iteration": variant.start_iteration,
        "iteration": iteration,
        "relative_iteration": iteration - variant.start_iteration,
        "stage": stage_name(variant, iteration),
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": point_count_at_iteration(run_dir, iteration),
        "train_time_s": train_time if iteration == variant.eval_iterations[-1] else None,
        "run_dir": str(run_dir),
        "source_run_dir": str(source_run_dir) if source_run_dir is not None else "",
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
    by_key: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["scene"]))
        by_key.setdefault(key, {})[int(row["iteration"])] = row

    for (variant_key, scene), scene_rows in sorted(by_key.items()):
        start_iteration = int(next(iter(scene_rows.values()))["start_iteration"])
        start = scene_rows.get(start_iteration)
        if start is None:
            continue
        for iteration in sorted(value for value in scene_rows if value > start_iteration):
            target = scene_rows[iteration]
            comparisons.append(
                {
                    "variant": variant_key,
                    "method": target["method"],
                    "mode": target["mode"],
                    "scene": scene,
                    "start_iteration": start_iteration,
                    "iteration": iteration,
                    "relative_iteration": iteration - start_iteration,
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
    keys = sorted({(str(row["variant"]), int(row["relative_iteration"])) for row in comparisons})
    for variant_key, relative_iteration in keys:
        subset = [
            row
            for row in comparisons
            if row["variant"] == variant_key and int(row["relative_iteration"]) == relative_iteration
        ]
        if not subset:
            continue
        row = {
            "variant": variant_key,
            "method": subset[0]["method"],
            "mode": subset[0]["mode"],
            "relative_iteration": relative_iteration,
            "scene_count": len(subset),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            values = [float(item[key]) for item in subset]
            row["avg_" + key] = sum(values) / len(values)
        aggregate.append(row)
    return aggregate


def read_baseline30(path: Path, scenes: set[str]) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "mipnerf360":
                continue
            if row.get("scene") not in scenes:
                continue
            if int(row.get("iteration", "0")) != 30000:
                continue
            rows[str(row["scene"])] = row
    return rows


def build_baseline30_comparison(rows: list[dict[str, object]], baseline30_csv: Path) -> list[dict[str, object]]:
    scenes = {str(row["scene"]) for row in rows}
    baseline = read_baseline30(baseline30_csv, scenes)
    out = []
    for row in rows:
        if int(row["iteration"]) not in (30000, 35000):
            continue
        base = baseline.get(str(row["scene"]))
        if base is None:
            continue
        out.append(
            {
                "variant": row["variant"],
                "method": row["method"],
                "mode": row["mode"],
                "scene": row["scene"],
                "iteration": row["iteration"],
                "stage": row["stage"],
                "baseline_iteration": 30000,
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


def write_summaries(output_root: Path, rows: list[dict[str, object]], baseline30_csv: Path) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    comparisons = build_start_comparisons(rows)
    write_csv(output_root / "comparison_start_to_eval.csv", comparisons)
    (output_root / "comparison_start_to_eval.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    aggregate = build_aggregate(comparisons)
    write_csv(output_root / "aggregate_start_to_eval.csv", aggregate)
    (output_root / "aggregate_start_to_eval.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_comparison = build_baseline30_comparison(rows, baseline30_csv)
    write_csv(output_root / "comparison_vs_baseline30.csv", baseline_comparison)
    (output_root / "comparison_vs_baseline30.json").write_text(
        json.dumps(baseline_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0018 supplemental indoor controls.")
    parser.add_argument("--dataset-name", default="mipnerf360", choices=["mipnerf360"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--baseline30-csv", type=Path, default=Path(DEFAULT_BASELINE30_CSV))
    parser.add_argument("--source-template", default=DEFAULT_0017_TEMPLATE)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--iterations", type=int, default=35000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--no-scene-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    use_overrides = not args.no_scene_overrides

    for variant_key in args.variants:
        variant = VARIANTS[variant_key]
        for scene in args.scenes:
            scene_path = args.dataset_root / scene
            if not scene_path.exists():
                raise FileNotFoundError(scene_path)
            overrides = scene_overrides(args.dataset_name, scene, use_overrides)
            run_dir = args.output_root / scene / variant.run_name
            log_dir = args.output_root / scene / "logs" / variant.run_name
            source_run_dir = None
            print("[0018-control] {} {} eval {}".format(
                variant.key,
                scene,
                ",".join(str(iteration) for iteration in variant.eval_iterations),
            ), flush=True)

            if variant.mode == "fulltrain":
                train_fulltrain(scene_path, run_dir, log_dir, repo, args, overrides, variant)
            elif variant.mode == "continuation":
                template = variant.source_template or args.source_template
                source_run_dir = Path(template.format(scene=scene))
                if not source_run_dir.exists():
                    raise FileNotFoundError(source_run_dir)
                train_continuation(scene_path, run_dir, log_dir, repo, args, overrides, variant, source_run_dir)
            else:
                raise ValueError("Unsupported mode: {}".format(variant.mode))

            render_all_then_metrics(run_dir, log_dir, repo, overrides, variant.eval_iterations)
            _, train_time = parse_train_log(log_dir / "train.log")

            for iteration in variant.eval_iterations:
                rows.append(row_for(args.dataset_name, scene, variant, iteration, run_dir, train_time, source_run_dir))
            write_summaries(args.output_root, rows, args.baseline30_csv)

    write_summaries(args.output_root, rows, args.baseline30_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
