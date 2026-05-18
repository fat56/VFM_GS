#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")
GS_RE = re.compile(r"Gaussian number:\s*(\d+)")
TIME_RE = re.compile(r"Training time:\s*([0-9.]+)")

DEFAULT_METHOD = "large_res_fastgs_big_densify100"
DEFAULT_RUN_NAME = "fastgs_big_densify100_30k_r_auto"
DEFAULT_VFM_CACHE_BACKEND = "depth_anything_v2"
DEFAULT_VFM_CACHE_FEATURE = "depth"
DEFAULT_VFM_CACHE_STORAGE = "npz_uint8"
DEFAULT_VFM_CACHE_MAX_WIDTH = 1600
DEFAULT_VFM_CACHE_DEVICE = "cuda"

MIPNERF360_OVERRIDES = {
    "bicycle": [],
    "flowers": ["--dense", "0.005", "--grad_abs_thresh", "0.001"],
    "garden": ["--highfeature_lr", "0.02", "--loss_thresh", "0.06", "--grad_abs_thresh", "0.0003"],
    "stump": ["--dense", "0.004", "--grad_abs_thresh", "0.001"],
    "treehill": ["--dense", "0.01", "--grad_abs_thresh", "0.0018"],
    "room": ["--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0004"],
    "counter": ["--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0004"],
    "kitchen": ["--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0002"],
    "bonsai": ["--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0002"],
}

DB_OVERRIDES = {
    "playroom": ["--highfeature_lr", "0.0015", "--dense", "0.003", "--mult", "0.7", "--grad_abs_thresh", "0.0005"],
    "drjohnson": [
        "--highfeature_lr",
        "0.0025",
        "--lowfeature_lr",
        "0.0005",
        "--grad_abs_thresh",
        "0.0005",
        "--dense",
        "0.005",
        "--mult",
        "0.7",
    ],
}

TANDT_OVERRIDES = {
    "truck": ["--highfeature_lr", "0.04", "--grad_abs_thresh", "0.0004", "--mult", "0.7"],
    "train": ["--highfeature_lr", "0.042", "--grad_abs_thresh", "0.0004", "--dense", "0.015", "--mult", "0.7"],
}


def run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ {}\n\n".format(" ".join(cmd)))
        handle.flush()
        proc_env = None
        if env is not None:
            proc_env = os.environ.copy()
            proc_env.update(env)
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=proc_env)
    return proc.returncode


def require_success(code: int, log_path: Path) -> None:
    if code != 0:
        raise RuntimeError("Command failed with exit code {}. See {}".format(code, log_path))


def parse_train_log(log_path: Path) -> tuple[int | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gs_matches = GS_RE.findall(text)
    time_matches = TIME_RE.findall(text)
    gs_num = int(gs_matches[-1]) if gs_matches else None
    train_time = float(time_matches[-1]) if time_matches else None
    return gs_num, train_time


def parse_metrics(run_dir: Path) -> dict[str, float | None]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {key: None for key in METRIC_KEYS}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        return {key: None for key in METRIC_KEYS}
    method = sorted(data.keys())[-1]
    return {key: data.get(method, {}).get(key) for key in METRIC_KEYS}


def is_metrics_complete(run_dir: Path) -> bool:
    metrics = parse_metrics(run_dir)
    return all(metrics[key] is not None for key in METRIC_KEYS)


def latest_point_count(run_dir: Path) -> int | None:
    point_cloud_dir = run_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None
    iterations = []
    for child in point_cloud_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iterations.append((int(child.name.split("_", 1)[1]), child))
        except ValueError:
            continue
    if not iterations:
        return None
    latest = sorted(iterations)[-1][1] / "point_cloud.ply"
    if not latest.exists():
        return None
    with latest.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    return None


def scene_overrides(dataset: str, scene: str, enabled: bool) -> list[str]:
    if not enabled:
        return []
    tables = {
        "mipnerf360": MIPNERF360_OVERRIDES,
        "db": DB_OVERRIDES,
        "tandt": TANDT_OVERRIDES,
    }
    return list(tables.get(dataset, {}).get(scene, []))


def build_vfm_cache(scene_path: Path, scene: str, args: argparse.Namespace, repo: Path, log_dir: Path, cache_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        return

    build_cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.build_vfm_cache",
        "-s",
        str(scene_path),
        "-i",
        args.train_images,
        "-o",
        str(cache_dir),
        "--backend",
        args.vfm_cache_backend,
        "--max_width",
        str(args.vfm_cache_max_width),
        "--device",
        args.vfm_cache_device,
        "--depth_anything_feature",
        args.vfm_cache_feature,
        "--storage",
        args.vfm_cache_storage,
    ]
    require_success(
        run_command(build_cmd, log_dir / "build_vfm_cache.log", repo, env={"HF_HUB_DISABLE_XET": "1"}),
        log_dir / "build_vfm_cache.log",
    )

    validate_cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.validate_vfm_cache",
        "-c",
        str(cache_dir),
        "-s",
        str(scene_path),
        "-i",
        args.train_images,
        "--backend",
        args.vfm_cache_backend,
    ]
    require_success(
        run_command(validate_cmd, log_dir / "validate_vfm_cache.log", repo, env={"HF_HUB_DISABLE_XET": "1"}),
        log_dir / "validate_vfm_cache.log",
    )


def option_value(options: list[str], name: str) -> str | None:
    try:
        return options[options.index(name) + 1]
    except (ValueError, IndexError):
        return None


def train_baseline(
    scene_path: Path,
    scene: str,
    args: argparse.Namespace,
    repo: Path,
    run_dir: Path,
    log_dir: Path,
    overrides: list[str],
    cache_dir: Path | None = None,
) -> None:
    if run_dir.exists() and latest_point_count(run_dir) is not None:
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
    ]
    if args.config is not None:
        cmd.extend(["--config", args.config])
    cmd.extend([
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
        str(args.iterations),
        "--checkpoint_iterations",
        str(args.iterations),
        "--densification_interval",
        str(args.densification_interval),
        "--optimizer_type",
        "default",
        "-r",
        str(args.resolution),
    ])
    if cache_dir is not None:
        cmd.extend(["--vfm_cache_dir", str(cache_dir)])
    cmd.extend(overrides)
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def render_and_metrics(run_dir: Path, log_dir: Path, repo: Path, overrides: list[str]) -> None:
    if not (run_dir / "test").exists():
        cmd = [
            "uv",
            "run",
            "--active",
            "python",
            "-m",
            "vfm_gs.cli.render",
            "-m",
            str(run_dir),
            "--iteration",
            "-1",
            "--skip_train",
            "--quiet",
        ]
        mult = option_value(overrides, "--mult")
        if mult is not None:
            cmd.extend(["--mult", mult])
        require_success(run_command(cmd, log_dir / "render.log", repo), log_dir / "render.log")
    if not is_metrics_complete(run_dir):
        cmd = [
            "uv",
            "run",
            "--active",
            "python",
            "-m",
            "vfm_gs.cli.metrics",
            "-m",
            str(run_dir),
        ]
        require_success(run_command(cmd, log_dir / "metrics.log", repo), log_dir / "metrics.log")


def collect_row(dataset: str, scene: str, method: str, run_dir: Path, log_dir: Path) -> dict[str, object]:
    gs_num, train_time = parse_train_log(log_dir / "train.log")
    if gs_num is None:
        gs_num = latest_point_count(run_dir)
    metrics = parse_metrics(run_dir)
    return {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "psnr": metrics["PSNR"],
        "ssim": metrics["SSIM"],
        "lpips": metrics["LPIPS"],
        "gs_num": gs_num,
        "train_time_s": train_time,
        "run_dir": str(run_dir),
    }


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "scene",
        "method",
        "psnr",
        "ssim",
        "lpips",
        "gs_num",
        "train_time_s",
        "run_dir",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    averages = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        avg = {"method": method, "scene_count": len(method_rows)}
        for source_key, target_key in [
            ("psnr", "avg_psnr"),
            ("ssim", "avg_ssim"),
            ("lpips", "avg_lpips"),
            ("gs_num", "avg_gs_num"),
            ("train_time_s", "avg_train_time_s"),
        ]:
            values = [float(row[source_key]) for row in method_rows if row[source_key] not in (None, "")]
            avg[target_key] = sum(values) / len(values) if values else None
        averages.append(avg)
    (output_dir / "averages.json").write_text(json.dumps(averages, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastGS scene evaluations and batch experiments.")
    parser.add_argument("--dataset-name", required=True, choices=["mipnerf360", "db", "tandt"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--config", default=None)
    parser.add_argument("--vfm-cache-template", default=None)
    parser.add_argument("--vfm-cache-backend", default=DEFAULT_VFM_CACHE_BACKEND)
    parser.add_argument("--vfm-cache-feature", default=DEFAULT_VFM_CACHE_FEATURE)
    parser.add_argument("--vfm-cache-storage", default=DEFAULT_VFM_CACHE_STORAGE)
    parser.add_argument("--vfm-cache-max-width", type=int, default=DEFAULT_VFM_CACHE_MAX_WIDTH)
    parser.add_argument("--vfm-cache-device", default=DEFAULT_VFM_CACHE_DEVICE)
    parser.add_argument("--no-scene-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    use_overrides = not args.no_scene_overrides

    for scene in args.scenes:
        scene_dir = args.dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)
        overrides = scene_overrides(args.dataset_name, scene, use_overrides)
        run_dir = args.output_root / scene / args.run_name
        log_dir = args.output_root / scene / "logs" / args.run_name
        cache_dir = None
        if args.vfm_cache_template:
            cache_dir = Path(args.vfm_cache_template.format(scene=scene))
            build_vfm_cache(scene_dir, scene, args, repo, log_dir, cache_dir)

        print("[{}] {} train/render/metrics".format(scene, args.method_name), flush=True)
        train_baseline(scene_dir, scene, args, repo, run_dir, log_dir, overrides, cache_dir=cache_dir)
        render_and_metrics(run_dir, log_dir, repo, overrides)
        rows.append(collect_row(args.dataset_name, scene, args.method_name, run_dir, log_dir))
        write_summary(rows, args.output_root)

    write_summary(rows, args.output_root)
    print("Wrote {}".format(args.output_root / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
