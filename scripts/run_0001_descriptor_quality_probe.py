#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = ("PSNR", "SSIM", "LPIPS")
GS_RE = re.compile(r"Gaussian number:\s*(\d+)")
TIME_RE = re.compile(r"Training time:\s*([0-9.]+)")


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


def latest_point_count(run_dir: Path) -> int | None:
    point_cloud_dir = run_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None
    iterations = []
    for child in point_cloud_dir.iterdir():
        if child.is_dir() and child.name.startswith("iteration_"):
            try:
                iterations.append((int(child.name.split("_", 1)[1]), child))
            except ValueError:
                pass
    if not iterations:
        return None
    ply_path = sorted(iterations)[-1][1] / "point_cloud.ply"
    if not ply_path.exists():
        return None
    with ply_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    return None


def parse_metrics(run_dir: Path) -> dict[str, float | None]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {key: None for key in METRIC_KEYS}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        return {key: None for key in METRIC_KEYS}
    method = sorted(data.keys())[-1]
    return {key: data.get(method, {}).get(key) for key in METRIC_KEYS}


def metrics_complete(run_dir: Path) -> bool:
    metrics = parse_metrics(run_dir)
    return all(metrics[key] is not None for key in METRIC_KEYS)


def cache_dir(scene: str, args: argparse.Namespace) -> Path:
    return args.cache_root / "{}_{}".format(scene, args.dino_backend)


def build_cache(scene_dir: Path, scene: str, args: argparse.Namespace, repo: Path, log_dir: Path) -> None:
    out_dir = cache_dir(scene, args)
    if (out_dir / "manifest.json").exists():
        return
    cmd = [
        "uv",
        "run",
        "--active",
        "python",
        "-m",
        "vfm_gs.cli.build_vfm_cache",
        "-s",
        str(scene_dir),
        "-i",
        args.cache_images,
        "-o",
        str(out_dir),
        "--backend",
        args.dino_backend,
        "--dinov2_repo",
        args.dinov2_repo,
        "--max_width",
        str(args.cache_max_width),
        "--storage",
        args.cache_storage,
    ]
    require_success(run_command(cmd, log_dir / "build_cache.log", repo), log_dir / "build_cache.log")


def validate_cache(scene_dir: Path, scene: str, args: argparse.Namespace, repo: Path, log_dir: Path) -> None:
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
        str(cache_dir(scene, args)),
        "-s",
        str(scene_dir),
        "-i",
        args.cache_images,
        "--backend",
        args.dino_backend,
    ]
    require_success(run_command(cmd, log_dir / "validate_cache.log", repo), log_dir / "validate_cache.log")
    marker.write_text("ok\n", encoding="utf-8")


def train_baseline(scene_dir: Path, run_dir: Path, log_dir: Path, args: argparse.Namespace, repo: Path) -> None:
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
        str(scene_dir),
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
        "-r",
        str(args.resolution),
    ]
    require_success(run_command(cmd, log_dir / "train.log", repo), log_dir / "train.log")


def train_descriptor(
    scene_dir: Path,
    scene: str,
    run_dir: Path,
    log_dir: Path,
    args: argparse.Namespace,
    repo: Path,
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
        "fastgs_baseline",
        "--config",
        args.config,
        "-s",
        str(scene_dir),
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
        "--vfm_cache_dir",
        str(cache_dir(scene, args)),
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
    if not metrics_complete(run_dir):
        cmd = ["uv", "run", "--active", "python", "-m", "vfm_gs.cli.metrics", "-m", str(run_dir)]
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


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "scene", "method", "psnr", "ssim", "lpips", "gs_num", "train_time_s", "run_dir"]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

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
    (output_root / "averages.json").write_text(json.dumps(averages, indent=2, ensure_ascii=False), encoding="utf-8")

    baseline_rows = {(row["scene"], row["method"]): row for row in rows if row["method"] == args_baseline_method()}
    comparisons = []
    for row in rows:
        if row["method"] == args_baseline_method():
            continue
        ref = baseline_rows.get((row["scene"], args_baseline_method()))
        if ref is None:
            continue
        comparison = {
            "dataset": row["dataset"],
            "scene": row["scene"],
            "method": row["method"],
            "reference": args_baseline_method(),
        }
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            comparison["delta_{}".format(key)] = float(row[key]) - float(ref[key])
        comparisons.append(comparison)
    if comparisons:
        fields = [
            "dataset",
            "scene",
            "method",
            "reference",
            "delta_psnr",
            "delta_ssim",
            "delta_lpips",
            "delta_gs_num",
            "delta_train_time_s",
        ]
        with (output_root / "comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparisons)
        (output_root / "comparisons.json").write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")


def args_baseline_method() -> str:
    return "fastgs_densify100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0001 DINO descriptor quality probes.")
    parser.add_argument("--dataset-name", default="mipnerf360")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, default=Path("output/0001/descriptor_densify_only_probe"))
    parser.add_argument("--scenes", nargs="+", default=["bicycle", "garden", "stump", "bonsai"])
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--cache-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--cache-root", type=Path, default=Path("output/0001/vfm_cache"))
    parser.add_argument("--cache-max-width", type=int, default=224)
    parser.add_argument("--cache-storage", default="npy_float16")
    parser.add_argument("--dino-backend", default="dinov2_vits14")
    parser.add_argument("--dinov2-repo", default="output/0001/external/dinov2")
    parser.add_argument("--config", default="configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only.yaml")
    parser.add_argument("--descriptor-method", default="dinov2_descriptor_densify_only")
    parser.add_argument("--baseline-run-name", default="fastgs_densify100_30k_r8")
    parser.add_argument("--descriptor-run-name", default="vfm_dinov2_descriptor_densify_only_30k_r8")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    global args
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []

    for scene in args.scenes:
        scene_dir = args.dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)

        print("[{}] descriptor quality probe".format(scene), flush=True)
        cache_log_dir = args.output_root / scene / "logs" / "cache"
        build_cache(scene_dir, scene, args, repo, cache_log_dir)
        validate_cache(scene_dir, scene, args, repo, cache_log_dir)

        if not args.skip_baseline:
            baseline_run = args.output_root / scene / args.baseline_run_name
            baseline_logs = args.output_root / scene / "logs" / args.baseline_run_name
            train_baseline(scene_dir, baseline_run, baseline_logs, args, repo)
            render_and_metrics(baseline_run, baseline_logs, repo)
            rows.append(collect_row(args.dataset_name, scene, args_baseline_method(), baseline_run, baseline_logs))

        descriptor_run = args.output_root / scene / args.descriptor_run_name
        descriptor_logs = args.output_root / scene / "logs" / args.descriptor_run_name
        train_descriptor(scene_dir, scene, descriptor_run, descriptor_logs, args, repo)
        render_and_metrics(descriptor_run, descriptor_logs, repo)
        rows.append(collect_row(args.dataset_name, scene, args.descriptor_method, descriptor_run, descriptor_logs))
        write_summary(rows, args.output_root)

    write_summary(rows, args.output_root)
    print("Wrote {}".format(args.output_root / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
