#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips")
OUTPUT_FIELDS = (
    "dataset",
    "scene",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "gs_num",
    "train_time_s",
    "phase0_psnr",
    "phase0_ssim",
    "phase0_lpips",
    "phase0_gs_num",
    "d_psnr_phase0",
    "d_ssim_phase0",
    "d_lpips_phase0",
    "d_gs_phase0",
    "curve_psnr",
    "curve_ssim",
    "curve_lpips",
    "curve_gs_num",
    "d_psnr_curve",
    "d_ssim_curve",
    "d_lpips_curve",
    "d_gs_curve",
    "auto_psnr",
    "auto_ssim",
    "auto_lpips",
    "auto_gs_num",
    "d_psnr_auto",
    "d_ssim_auto",
    "d_lpips_auto",
    "d_gs_auto",
    "run_dir",
    "source_summary",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_summary_rows(paths: list[Path]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    rows_by_scene: dict[str, dict[str, str]] = {}
    duplicates: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            scene = row["scene"]
            row = dict(row)
            row["source_summary"] = str(path)
            if scene in rows_by_scene:
                duplicates.append(
                    {
                        "scene": scene,
                        "old_source": rows_by_scene[scene].get("source_summary", ""),
                        "new_source": str(path),
                    }
                )
            rows_by_scene[scene] = row
    return rows_by_scene, duplicates


def load_phase0_baseline(path: Path) -> dict[str, dict[str, str]]:
    return {row["scene"]: row for row in read_csv(path)}


def load_curve_baseline(path: Path, iteration: int) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if int(row["iteration"]) != iteration:
            continue
        rows[row["scene"]] = row
    return rows


def add_baseline_delta(
    out: dict[str, Any],
    row: dict[str, str],
    baseline: dict[str, str],
    prefix: str,
    gs_key: str,
) -> None:
    for key in METRIC_KEYS:
        out[f"{prefix}_{key}"] = as_float(baseline, key)
        out[f"d_{key}_{prefix}"] = as_float(row, key) - as_float(baseline, key)
    out[f"{prefix}_gs_num"] = as_int(baseline, gs_key)
    out[f"d_gs_{prefix}"] = as_int(row, "gs_num") - as_int(baseline, gs_key)


def build_comparison_rows(
    experiment_rows: dict[str, dict[str, str]],
    phase0: dict[str, dict[str, str]],
    curve: dict[str, dict[str, str]],
    auto: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, row in sorted(experiment_rows.items()):
        if scene not in phase0 or scene not in curve or scene not in auto:
            continue
        out: dict[str, Any] = {
            "dataset": row["dataset"],
            "scene": scene,
            "method": row["method"],
            "psnr": as_float(row, "psnr"),
            "ssim": as_float(row, "ssim"),
            "lpips": as_float(row, "lpips"),
            "gs_num": as_int(row, "gs_num"),
            "train_time_s": as_float(row, "train_time_s"),
            "run_dir": row.get("run_dir", ""),
            "source_summary": row.get("source_summary", ""),
        }
        add_baseline_delta(out, row, phase0[scene], "phase0", "gs_num")
        add_baseline_delta(out, row, curve[scene], "curve", "gs_num")
        add_baseline_delta(out, row, auto[scene], "auto", "gs_num")
        rows.append(out)
    return rows


def summarize(rows: list[dict[str, Any]], duplicates: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "scene_count": len(rows),
        "avg_psnr": mean([float(row["psnr"]) for row in rows]),
        "avg_ssim": mean([float(row["ssim"]) for row in rows]),
        "avg_lpips": mean([float(row["lpips"]) for row in rows]),
        "avg_gs_num": mean([float(row["gs_num"]) for row in rows]),
        "avg_d_psnr_phase0": mean([float(row["d_psnr_phase0"]) for row in rows]),
        "avg_d_ssim_phase0": mean([float(row["d_ssim_phase0"]) for row in rows]),
        "avg_d_lpips_phase0": mean([float(row["d_lpips_phase0"]) for row in rows]),
        "avg_d_gs_phase0": mean([float(row["d_gs_phase0"]) for row in rows]),
        "avg_d_psnr_curve": mean([float(row["d_psnr_curve"]) for row in rows]),
        "avg_d_ssim_curve": mean([float(row["d_ssim_curve"]) for row in rows]),
        "avg_d_lpips_curve": mean([float(row["d_lpips_curve"]) for row in rows]),
        "avg_d_gs_curve": mean([float(row["d_gs_curve"]) for row in rows]),
        "avg_d_psnr_auto": mean([float(row["d_psnr_auto"]) for row in rows]),
        "avg_d_ssim_auto": mean([float(row["d_ssim_auto"]) for row in rows]),
        "avg_d_lpips_auto": mean([float(row["d_lpips_auto"]) for row in rows]),
        "avg_d_gs_auto": mean([float(row["d_gs_auto"]) for row in rows]),
        "duplicates": duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment 0009 residual-orientation comparison tables.")
    parser.add_argument(
        "--summary",
        nargs="+",
        type=Path,
        default=[
            Path("output/0009/residual_orientation_protect_pilot/mip_g0/summary.csv"),
            Path("output/0009/residual_orientation_protect_pilot/mip_g1/summary.csv"),
            Path("output/0009/residual_orientation_protect_full_missing/mip_g0/summary.csv"),
            Path("output/0009/residual_orientation_protect_full_missing/mip_g1/summary.csv"),
        ],
        help="Experiment summary.csv files to merge.",
    )
    parser.add_argument(
        "--phase0-baseline",
        type=Path,
        default=Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv"),
    )
    parser.add_argument(
        "--curve-baseline",
        type=Path,
        default=Path("output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv"),
    )
    parser.add_argument(
        "--auto-baseline",
        type=Path,
        default=Path(
            "output/0002/depth_anything_depth_prior_prune_protect_auto_topk_full/"
            "mipnerf360_combined/summary.csv"
        ),
    )
    parser.add_argument("--curve-iteration", type=int, default=30000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/0009/residual_orientation_protect_comparison"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_rows, duplicates = load_summary_rows(args.summary)
    if not experiment_rows:
        raise ValueError("No experiment rows loaded")

    phase0 = load_phase0_baseline(args.phase0_baseline)
    curve = load_curve_baseline(args.curve_baseline, args.curve_iteration)
    auto = load_phase0_baseline(args.auto_baseline)
    rows = build_comparison_rows(experiment_rows, phase0, curve, auto)
    summary = summarize(rows, duplicates)

    write_csv(args.output_dir / "comparison.csv", rows, OUTPUT_FIELDS)
    write_json(args.output_dir / "summary.json", summary)

    print("Wrote {}".format(args.output_dir / "comparison.csv"))
    print("Wrote {}".format(args.output_dir / "summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
