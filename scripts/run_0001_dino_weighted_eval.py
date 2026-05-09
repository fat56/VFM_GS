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
DEFAULT_METHOD = "dinov2_token_edge_weighted_i050"
DEFAULT_RUN_NAME = "vfm_dinov2_token_edge_topk025_weighted_i050_30k_r8"


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


def build_dino_cache(scene_path: Path, args: argparse.Namespace, repo: Path, cache_dir: Path, log_dir: Path) -> None:
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
        str(scene_path),
        "-i",
        args.cache_images,
        "-o",
        str(cache_dir),
        "--backend",
        args.dino_backend,
        "--dinov2_repo",
        args.dinov2_repo,
        "--max_width",
        str(args.cache_max_width),
        "--storage",
        args.cache_storage,
    ]
    if args.project_token_edge:
        cmd.append("--project_token_edge")
    require_success(run_command(cmd, log_dir / "build_cache.log", repo), log_dir / "build_cache.log")


def validate_dino_cache(scene_path: Path, args: argparse.Namespace, repo: Path, cache_dir: Path, log_dir: Path) -> None:
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
        str(scene_path),
        "-i",
        args.cache_images,
        "--backend",
        args.dino_backend,
    ]
    require_success(run_command(cmd, log_dir / "validate_cache.log", repo), log_dir / "validate_cache.log")
    marker.write_text("ok\n", encoding="utf-8")


def reference_gs_for_scene(reference_rows: list[dict[str, object]], scene: str, method: str) -> int | None:
    for row in reference_rows:
        if str(row.get("scene")) != scene or str(row.get("method")) != method:
            continue
        value = row.get("gs_num")
        if value in (None, ""):
            return None
        return int(float(value))
    return None


def train_candidate(
    scene_path: Path,
    args: argparse.Namespace,
    repo: Path,
    run_dir: Path,
    log_dir: Path,
    cache_dir: Path,
    target: int | None,
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
        "--vfm_cache_dir",
        str(cache_dir),
        "-r",
        str(args.resolution),
    ]
    if target is not None:
        cmd.extend(["--target_gaussian_count", str(target)])
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


def collect_row(
    dataset: str,
    scene: str,
    method: str,
    run_dir: Path,
    log_dir: Path,
    target_gaussian_count: int | None,
) -> dict[str, object]:
    gs_num, train_time = parse_train_log(log_dir / "train.log")
    if gs_num is None:
        gs_num = latest_point_count(run_dir)
    metrics = parse_metrics(run_dir)
    return {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "target_gaussian_count": target_gaussian_count,
        "psnr": metrics["PSNR"],
        "ssim": metrics["SSIM"],
        "lpips": metrics["LPIPS"],
        "gs_num": gs_num,
        "train_time_s": train_time,
        "run_dir": str(run_dir),
    }


def read_reference_rows(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def write_summary(
    rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    output_dir: Path,
    comparison_methods: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
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

    if not reference_rows:
        return

    comparisons = []
    by_scene_method = {
        (str(row.get("scene")), str(row.get("method"))): row
        for row in reference_rows
    }
    for row in rows:
        for ref_method in comparison_methods:
            ref = by_scene_method.get((str(row["scene"]), ref_method))
            if ref is None:
                continue
            comparison = {
                "dataset": row["dataset"],
                "scene": row["scene"],
                "method": row["method"],
                "reference": ref_method,
            }
            for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
                value = to_float(row.get(key))
                ref_value = to_float(ref.get(key))
                comparison["delta_{}".format(key)] = None if value is None or ref_value is None else value - ref_value
            comparisons.append(comparison)
    if comparisons:
        comparison_fields = [
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
        with (output_dir / "comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(comparisons)
        (output_dir / "comparisons.json").write_text(
            json.dumps(comparisons, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0001 DINO token-edge candidate evaluation.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--cache-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--cache-max-width", type=int, default=224)
    parser.add_argument("--cache-storage", default="npy_float16")
    parser.add_argument("--project-token-edge", action="store_true")
    parser.add_argument("--dino-backend", default="dinov2_vits14")
    parser.add_argument("--dinov2-repo", default="output/0001/external/dinov2")
    parser.add_argument("--config", default="configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050.yaml")
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--reference-summary", type=Path, default=None)
    parser.add_argument("--comparison-methods", nargs="+", default=["baseline", "cached_edge_staged142"])
    parser.add_argument("--target-ratio-from-reference", type=float, default=0.0)
    parser.add_argument("--target-reference-method", default="baseline")
    return parser.parse_args()


def cache_dir_for_scene(scene: str, args: argparse.Namespace) -> Path:
    feature_suffix = "_token_edge_w{}".format(args.cache_max_width) if args.project_token_edge else ""
    if args.cache_root is not None:
        return args.cache_root / "{}_{}{}".format(scene, args.dino_backend, feature_suffix)
    return args.output_root / scene / "cache" / "{}{}_w{}".format(
        args.dino_backend,
        "_token_edge" if args.project_token_edge else "",
        args.cache_max_width,
    )


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    reference_rows = read_reference_rows(args.reference_summary)

    for scene in args.scenes:
        scene_dir = args.dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)

        run_dir = args.output_root / scene / args.run_name
        log_dir = args.output_root / scene / "logs" / args.run_name
        cache_dir = cache_dir_for_scene(scene, args)
        target = None
        if args.target_ratio_from_reference > 0:
            reference_gs = reference_gs_for_scene(reference_rows, scene, args.target_reference_method)
            if reference_gs is None:
                raise RuntimeError(
                    "Unable to determine reference Gaussian count for {} using method {}".format(
                        scene, args.target_reference_method
                    )
                )
            target = int(round(reference_gs * args.target_ratio_from_reference))

        print("[{}] {} cache/train/render/metrics".format(scene, args.method_name), flush=True)
        build_dino_cache(scene_dir, args, repo, cache_dir, log_dir)
        validate_dino_cache(scene_dir, args, repo, cache_dir, log_dir)
        train_candidate(scene_dir, args, repo, run_dir, log_dir, cache_dir, target)
        render_and_metrics(run_dir, log_dir, repo)
        rows.append(collect_row(args.dataset_name, scene, args.method_name, run_dir, log_dir, target))
        write_summary(rows, reference_rows, args.output_root, args.comparison_methods)

    write_summary(rows, reference_rows, args.output_root, args.comparison_methods)
    print("Wrote {}".format(args.output_root / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
