#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SCENES = [
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
]

METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")
GS_RE = re.compile(r"Gaussian number:\s*(\d+)")
TIME_RE = re.compile(r"Training time:\s*([0-9.]+)")
EDGE_RATIO = 1.42


def run_command(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ {}\n\n".format(" ".join(cmd)))
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True)
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


def train_baseline(scene: str, args: argparse.Namespace, repo: Path, run_dir: Path, log_dir: Path) -> None:
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
        "fastgs_baseline",
        "-s",
        str(args.dataset_root / scene),
        "-i",
        "images",
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
        "-r",
        str(args.resolution),
    ]
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def build_edge_cache(scene: str, args: argparse.Namespace, repo: Path, cache_dir: Path, log_dir: Path) -> None:
    if (cache_dir / "manifest.json").exists():
        return
    cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.build_vfm_cache",
        "-s",
        str(args.dataset_root / scene),
        "-i",
        "images_8",
        "-o",
        str(cache_dir),
        "--max_width",
        "640",
        "--storage",
        "npz_uint8",
    ]
    require_success(run_command(cmd, log_dir / "build_cache.log", repo), log_dir / "build_cache.log")


def validate_edge_cache(scene: str, args: argparse.Namespace, repo: Path, cache_dir: Path, log_dir: Path) -> None:
    marker = log_dir / "validate_cache.ok"
    if marker.exists():
        return
    cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.validate_vfm_cache",
        "-c",
        str(cache_dir),
        "-s",
        str(args.dataset_root / scene),
        "-i",
        "images_8",
        "--backend",
        "cached_edge_l1",
    ]
    require_success(run_command(cmd, log_dir / "validate_cache.log", repo), log_dir / "validate_cache.log")
    marker.write_text("ok\n", encoding="utf-8")


def train_edge(scene: str, args: argparse.Namespace, repo: Path, run_dir: Path, log_dir: Path, cache_dir: Path, target: int) -> None:
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
        "fastgs_baseline",
        "--config",
        "configs/experiments/0001_vfm_topology_cached_edge_compact.yaml",
        "-s",
        str(args.dataset_root / scene),
        "-i",
        "images",
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
        "--vfm_cache_dir",
        str(cache_dir),
        "--target_gaussian_count",
        str(target),
        "--target_gaussian_staged",
        "--target_gaussian_stage_margin",
        "1.10",
        "--target_gaussian_stage_interval",
        "500",
        "-r",
        str(args.resolution),
    ]
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def render_and_metrics(run_dir: Path, log_dir: Path, repo: Path) -> None:
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


def collect_row(scene: str, method: str, run_dir: Path, log_dir: Path, target: int | None = None) -> dict[str, object]:
    gs_num, train_time = parse_train_log(log_dir / "train.log")
    if gs_num is None:
        gs_num = latest_point_count(run_dir)
    metrics = parse_metrics(run_dir)
    row: dict[str, object] = {
        "scene": scene,
        "method": method,
        "target_gaussian_count": target or "",
        "psnr": metrics["PSNR"],
        "ssim": metrics["SSIM"],
        "lpips": metrics["LPIPS"],
        "gs_num": gs_num,
        "train_time_s": train_time,
        "run_dir": str(run_dir),
    }
    return row


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fields = [
        "scene",
        "method",
        "target_gaussian_count",
        "psnr",
        "ssim",
        "lpips",
        "gs_num",
        "train_time_s",
        "run_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
    parser = argparse.ArgumentParser(description="Run full MipNeRF360 baseline and cached-edge v1 evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, default=Path("output/0001/full_mipnerf360_v1"))
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        scene_dir = args.dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)

        baseline_run = args.output_root / scene / "baseline_30k_r8"
        baseline_logs = args.output_root / scene / "logs" / "baseline_30k_r8"
        edge_run = args.output_root / scene / "vfm_cached_edge_staged142_30k_r8"
        edge_logs = args.output_root / scene / "logs" / "vfm_cached_edge_staged142_30k_r8"
        cache_dir = args.output_root / scene / "cache" / "edge_u8"

        print("[{}] baseline train/render/metrics".format(scene), flush=True)
        train_baseline(scene, args, repo, baseline_run, baseline_logs)
        render_and_metrics(baseline_run, baseline_logs, repo)
        baseline_row = collect_row(scene, "baseline", baseline_run, baseline_logs)
        rows.append(baseline_row)

        baseline_gs = baseline_row["gs_num"]
        if baseline_gs in (None, ""):
            raise RuntimeError("Unable to determine baseline Gaussian count for {}".format(scene))
        edge_target = int(round(float(baseline_gs) * EDGE_RATIO))

        print("[{}] cached-edge cache/train/render/metrics target={}".format(scene, edge_target), flush=True)
        build_edge_cache(scene, args, repo, cache_dir, edge_logs)
        validate_edge_cache(scene, args, repo, cache_dir, edge_logs)
        train_edge(scene, args, repo, edge_run, edge_logs, cache_dir, edge_target)
        render_and_metrics(edge_run, edge_logs, repo)
        rows.append(collect_row(scene, "cached_edge_staged142", edge_run, edge_logs, target=edge_target))

        write_summary(rows, args.output_root)

    write_summary(rows, args.output_root)
    print("Wrote {}".format(args.output_root / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
