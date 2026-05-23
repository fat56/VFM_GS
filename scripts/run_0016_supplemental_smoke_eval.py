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


DEFAULT_BASELINE_TEMPLATE = (
    "output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/"
    "{scene}/fastgs_big_30k_curve_r_auto"
)
DEFAULT_DESCRIPTOR_CACHE_TEMPLATE = "output/0001/vfm_cache/{scene}_dinov2_vits14"
DEFAULT_TOKENEDGE_CACHE_TEMPLATE = "output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600"
DEFAULT_SCENES = ["kitchen", "flowers", "bonsai"]
RUN_NAME_TEMPLATE = "{method}_i{switch}_to_i{final}_2k_r_auto"


@dataclass(frozen=True)
class SmokeVariant:
    key: str
    method: str
    backend: str
    config: str
    cache_template: str
    start_iteration: int


VARIANTS: dict[str, SmokeVariant] = {
    "desc_clone_16k": SmokeVariant(
        key="desc_clone_16k",
        method="descriptor_clone_only_i16000_2k",
        backend="descriptor",
        config="configs/experiments/0016_smoke_descriptor_clone_only.yaml",
        cache_template=DEFAULT_DESCRIPTOR_CACHE_TEMPLATE,
        start_iteration=16000,
    ),
    "desc_split_16k": SmokeVariant(
        key="desc_split_16k",
        method="descriptor_split_only_i16000_2k",
        backend="descriptor",
        config="configs/experiments/0016_smoke_descriptor_split_only.yaml",
        cache_template=DEFAULT_DESCRIPTOR_CACHE_TEMPLATE,
        start_iteration=16000,
    ),
    "desc_clone_30k": SmokeVariant(
        key="desc_clone_30k",
        method="descriptor_clone_only_i30000_2k",
        backend="descriptor",
        config="configs/experiments/0016_smoke_descriptor_clone_only.yaml",
        cache_template=DEFAULT_DESCRIPTOR_CACHE_TEMPLATE,
        start_iteration=30000,
    ),
    "tokenedge_split_30k": SmokeVariant(
        key="tokenedge_split_30k",
        method="tokenedge_split_only_i30000_2k",
        backend="tokenedge",
        config="configs/experiments/0016_smoke_tokenedge_split_only.yaml",
        cache_template=DEFAULT_TOKENEDGE_CACHE_TEMPLATE,
        start_iteration=30000,
    ),
}


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


def ensure_metrics(run_dir: Path, log_dir: Path, repo: Path, overrides: list[str], iteration: int) -> None:
    metrics = read_metrics(run_dir, iteration)
    if all(metrics[key] is not None for key in ("PSNR", "SSIM", "LPIPS")):
        return
    render_iteration(run_dir, log_dir, repo, overrides, iteration)
    run_metrics(run_dir, log_dir, repo)


def train_followup(
    scene_path: Path,
    run_dir: Path,
    log_dir: Path,
    repo: Path,
    args: argparse.Namespace,
    overrides: list[str],
    variant: SmokeVariant,
    eval_iterations: list[int],
    cache_dir: Path,
) -> None:
    final_iteration = eval_iterations[-1]
    if point_count_at_iteration(run_dir, final_iteration) is not None:
        return

    save_args = [str(iteration) for iteration in eval_iterations[1:]]
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
        str(final_iteration),
        "--start_pointcloud_iteration",
        str(variant.start_iteration),
        "--test_iterations",
        *save_args,
        "--save_iterations",
        *save_args,
        "--checkpoint_iterations",
        str(final_iteration),
        "--densification_interval",
        str(args.densification_interval),
        "--densify_from_iter",
        str(variant.start_iteration),
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


def row_for(
    dataset: str,
    scene: str,
    variant: SmokeVariant,
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
        "variant": variant.key,
        "method": variant.method,
        "backend": variant.backend,
        "start_iteration": variant.start_iteration,
        "iteration": iteration,
        "relative_iteration": iteration - variant.start_iteration,
        "stage": stage,
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": point_count_at_iteration(run_dir, iteration),
        "train_time_s": train_time if stage == "end_2k" else None,
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


def write_summaries(output_root: Path, rows: list[dict[str, object]]) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    comparisons = []
    by_key: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["scene"]))
        by_key.setdefault(key, {})[str(row["stage"])] = row

    for (variant_key, scene), stages in sorted(by_key.items()):
        start = stages.get("start")
        if start is None:
            continue
        for target_stage in ("mid_1k", "end_2k"):
            target = stages.get(target_stage)
            if target is None:
                continue
            comparisons.append(
                {
                    "variant": variant_key,
                    "method": target["method"],
                    "backend": target["backend"],
                    "scene": scene,
                    "start_iteration": target["start_iteration"],
                    "iteration": target["iteration"],
                    "relative_iteration": target["relative_iteration"],
                    "stage": target_stage,
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
    write_csv(output_root / "comparison_start_to_eval.csv", comparisons)
    (output_root / "comparison_start_to_eval.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    aggregate = []
    for variant_key in sorted({row["variant"] for row in comparisons}):
        for relative_iteration in sorted({int(row["relative_iteration"]) for row in comparisons}):
            subset = [
                row
                for row in comparisons
                if row["variant"] == variant_key and int(row["relative_iteration"]) == relative_iteration
            ]
            if not subset:
                continue
            agg = {
                "variant": variant_key,
                "method": subset[0]["method"],
                "backend": subset[0]["backend"],
                "relative_iteration": relative_iteration,
                "scene_count": len(subset),
            }
            for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
                values = [float(row[key]) for row in subset]
                agg["avg_" + key] = sum(values) / len(values)
            aggregate.append(agg)
    write_csv(output_root / "aggregate_by_variant.csv", aggregate)
    (output_root / "aggregate_by_variant.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0016 supplemental 2k smoke continuations.")
    parser.add_argument("--dataset-name", default="mipnerf360", choices=["mipnerf360"])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS.keys()), default=sorted(VARIANTS.keys()))
    parser.add_argument("--baseline-template", default=DEFAULT_BASELINE_TEMPLATE)
    parser.add_argument("--followup-iterations", type=int, default=2000)
    parser.add_argument("--mid-iteration", type=int, default=1000)
    parser.add_argument("--train-images", default="images")
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
        smoke_variant = VARIANTS[variant_key]
        eval_offsets = [0, int(args.mid_iteration), int(args.followup_iterations)]
        eval_iterations = [smoke_variant.start_iteration + offset for offset in eval_offsets]
        for scene in args.scenes:
            scene_path = args.dataset_root / scene
            if not scene_path.exists():
                raise FileNotFoundError(scene_path)
            overrides = scene_overrides(args.dataset_name, scene, use_overrides)
            baseline_run_dir = Path(args.baseline_template.format(scene=scene))
            if not baseline_run_dir.exists():
                raise FileNotFoundError(baseline_run_dir)
            cache_dir = Path(smoke_variant.cache_template.format(scene=scene))
            if not (cache_dir / "manifest.json").exists():
                raise FileNotFoundError(cache_dir / "manifest.json")

            final_iteration = eval_iterations[-1]
            run_name = RUN_NAME_TEMPLATE.format(
                method=smoke_variant.method,
                switch=smoke_variant.start_iteration,
                final=final_iteration,
            )
            run_dir = args.output_root / smoke_variant.key / scene / run_name
            log_dir = args.output_root / smoke_variant.key / scene / "logs" / run_name
            print(
                "[0016-smoke] {} {} i{} -> i{} eval {}".format(
                    smoke_variant.key,
                    scene,
                    smoke_variant.start_iteration,
                    final_iteration,
                    ",".join(str(x) for x in eval_iterations),
                ),
                flush=True,
            )

            copy_start_state(baseline_run_dir, run_dir, smoke_variant.start_iteration)
            ensure_metrics(run_dir, log_dir, repo, overrides, smoke_variant.start_iteration)
            train_followup(
                scene_path,
                run_dir,
                log_dir,
                repo,
                args,
                overrides,
                smoke_variant,
                eval_iterations,
                cache_dir,
            )
            for iteration in eval_iterations[1:]:
                ensure_metrics(run_dir, log_dir, repo, overrides, iteration)
            _, train_time = parse_train_log(log_dir / "train.log")

            for stage, iteration in (
                ("start", eval_iterations[0]),
                ("mid_1k", eval_iterations[1]),
                ("end_2k", eval_iterations[2]),
            ):
                rows.append(
                    row_for(
                        args.dataset_name,
                        scene,
                        smoke_variant,
                        stage,
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
