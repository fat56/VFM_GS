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
DEFAULT_CACHE_TEMPLATE = "output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600"
DEFAULT_CONFIG = "configs/experiments/0016_tokenedge_split_only_after16k.yaml"
RUN_NAME_TEMPLATE = "tokenedge_split_only_i{switch}_to_i{final}_5k_r_auto"


def read_metrics(run_dir: Path, iteration: int) -> dict[str, float | None]:
    path = run_dir / "results.json"
    if not path.exists():
        return {"PSNR": None, "SSIM": None, "LPIPS": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("ours_{}".format(iteration), {"PSNR": None, "SSIM": None, "LPIPS": None})


def parse_train_log(log_path: Path) -> tuple[int | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gs_matches = re.findall(r"Gaussian number:\s*(\d+)", text)
    time_matches = re.findall(r"Training time:\s*([0-9.]+)", text)
    gs_num = int(gs_matches[-1]) if gs_matches else None
    train_time = float(time_matches[-1]) if time_matches else None
    return gs_num, train_time


def copy_start_state(baseline_run_dir: Path, run_dir: Path, switch_iteration: int) -> None:
    source_ply = baseline_run_dir / "point_cloud" / "iteration_{}".format(switch_iteration) / "point_cloud.ply"
    if not source_ply.exists():
        raise FileNotFoundError("Missing baseline point cloud: {}".format(source_ply))

    target_dir = run_dir / "point_cloud" / "iteration_{}".format(switch_iteration)
    target_ply = target_dir / "point_cloud.ply"
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_ply.exists():
        shutil.copy2(source_ply, target_ply)

    for name in ("cfg_args", "cameras.json", "input.ply"):
        source = baseline_run_dir / name
        target = run_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def ensure_start_metrics(run_dir: Path, log_dir: Path, repo: Path, overrides: list[str], switch_iteration: int) -> None:
    metrics = read_metrics(run_dir, switch_iteration)
    if all(metrics[key] is not None for key in ("PSNR", "SSIM", "LPIPS")):
        return
    render_iteration(run_dir, log_dir, repo, overrides, switch_iteration)
    run_metrics(run_dir, log_dir, repo)


def train_followup(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    switch_iteration: int,
    final_iteration: int,
    cache_dir: Path,
) -> None:
    if point_count_at_iteration(run_dir, final_iteration) is not None:
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
        str(switch_iteration),
        "--test_iterations",
        str(final_iteration),
        "--save_iterations",
        str(final_iteration),
        "--checkpoint_iterations",
        str(final_iteration),
        "--densification_interval",
        str(args.densification_interval),
        "--densify_from_iter",
        str(switch_iteration),
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


def ensure_final_metrics(run_dir: Path, log_dir: Path, repo: Path, overrides: list[str], final_iteration: int) -> None:
    metrics = read_metrics(run_dir, final_iteration)
    if all(metrics[key] is not None for key in ("PSNR", "SSIM", "LPIPS")):
        return
    render_iteration(run_dir, log_dir, repo, overrides, final_iteration)
    run_metrics(run_dir, log_dir, repo)


def row_for(
    dataset: str,
    scene: str,
    method: str,
    switch_iteration: int,
    final_iteration: int,
    stage: str,
    iteration: int,
    run_dir: Path,
    train_time: float | None,
    cache_dir: Path,
) -> dict[str, object]:
    metrics = read_metrics(run_dir, iteration)
    return {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "switch_iteration": switch_iteration,
        "final_iteration": final_iteration,
        "stage": stage,
        "iteration": iteration,
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": point_count_at_iteration(run_dir, iteration),
        "train_time_s": train_time if stage == "tokenedge_end" else None,
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(output_root: Path, rows: list[dict[str, object]]) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    comparisons = []
    by_key: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scene"]), int(row["switch_iteration"]))
        by_key.setdefault(key, {})[str(row["stage"])] = row
    for (scene, switch_iteration), stages in sorted(by_key.items()):
        start = stages.get("baseline_start")
        final = stages.get("tokenedge_end")
        if start is None or final is None:
            continue
        comparisons.append(
            {
                "scene": scene,
                "switch_iteration": switch_iteration,
                "final_iteration": final["final_iteration"],
                "delta_psnr": float(final["psnr"]) - float(start["psnr"]),
                "delta_ssim": float(final["ssim"]) - float(start["ssim"]),
                "delta_lpips": float(final["lpips"]) - float(start["lpips"]),
                "delta_gs_num": int(final["gs_num"]) - int(start["gs_num"]),
                "start_psnr": start["psnr"],
                "final_psnr": final["psnr"],
                "start_ssim": start["ssim"],
                "final_ssim": final["ssim"],
                "start_lpips": start["lpips"],
                "final_lpips": final["lpips"],
                "start_gs_num": start["gs_num"],
                "final_gs_num": final["gs_num"],
                "run_dir": final["run_dir"],
            }
        )
    write_csv(output_root / "comparison_start_to_final.csv", comparisons)
    (output_root / "comparison_start_to_final.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0016 token-edge split-only 5k continuations.")
    parser.add_argument("--dataset-name", default="mipnerf360", choices=["mipnerf360"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=MIPNERF360_SCENES)
    parser.add_argument("--switch-iterations", nargs="+", type=int, default=[16000, 18000, 20000])
    parser.add_argument("--followup-iterations", type=int, default=5000)
    parser.add_argument("--baseline-template", default=DEFAULT_BASELINE_TEMPLATE)
    parser.add_argument("--cache-template", default=DEFAULT_CACHE_TEMPLATE)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--method-name", default="tokenedge_split_only_5k")
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
        baseline_run_dir = Path(args.baseline_template.format(scene=scene))
        if not baseline_run_dir.exists():
            raise FileNotFoundError(baseline_run_dir)
        cache_dir = Path(args.cache_template.format(scene=scene))
        if not (cache_dir / "manifest.json").exists():
            raise FileNotFoundError(cache_dir / "manifest.json")

        for switch_iteration in args.switch_iterations:
            final_iteration = int(switch_iteration) + int(args.followup_iterations)
            run_name = RUN_NAME_TEMPLATE.format(switch=switch_iteration, final=final_iteration)
            run_dir = args.output_root / scene / run_name
            log_dir = args.output_root / scene / "logs" / run_name
            print("[0016] {} i{} -> i{}".format(scene, switch_iteration, final_iteration), flush=True)

            copy_start_state(baseline_run_dir, run_dir, switch_iteration)
            ensure_start_metrics(run_dir, log_dir, repo, overrides, switch_iteration)
            train_followup(
                scene_path,
                run_dir,
                log_dir,
                repo,
                args,
                overrides,
                switch_iteration,
                final_iteration,
                cache_dir,
            )
            ensure_final_metrics(run_dir, log_dir, repo, overrides, final_iteration)
            _, train_time = parse_train_log(log_dir / "train.log")

            rows.append(
                row_for(
                    args.dataset_name,
                    scene,
                    args.method_name,
                    switch_iteration,
                    final_iteration,
                    "baseline_start",
                    switch_iteration,
                    run_dir,
                    train_time,
                    cache_dir,
                )
            )
            rows.append(
                row_for(
                    args.dataset_name,
                    scene,
                    args.method_name,
                    switch_iteration,
                    final_iteration,
                    "tokenedge_end",
                    final_iteration,
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
