#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
QUALITY_WEIGHTS = {"psnr": 1.0, "ssim": 20.0, "lpips": 5.0}
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04

BASELINE = Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv")
METHODS = {
    "depth_auto_topk": Path("output/0002/depth_anything_depth_prior_prune_protect_auto_topk_full/mipnerf360_combined/summary.csv"),
    "residual_orientation": Path("output/0009/residual_orientation_protect_full/comparison/comparison.csv"),
    "dino_i050_full": Path("output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary.csv"),
    "dino_i050_until8000": Path("output/0010/descriptor_i050_until8000_full9_combined/summary.csv"),
    "dino_i050_until4000": Path("output/0011/descriptor_i050_until4000/mipnerf360_combined/summary.csv"),
}
OUT_DIR = Path("output/0013/cross_method_scene_oracle")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def quality_gain(row: dict[str, object], ref: dict[str, object]) -> float:
    d_psnr = as_float(row, "psnr") - as_float(ref, "psnr")
    d_ssim = as_float(row, "ssim") - as_float(ref, "ssim")
    d_lpips = as_float(row, "lpips") - as_float(ref, "lpips")
    return QUALITY_WEIGHTS["psnr"] * d_psnr + QUALITY_WEIGHTS["ssim"] * d_ssim - QUALITY_WEIGHTS["lpips"] * d_lpips


def gs_penalty(delta_gs: float) -> float:
    growth = max(0.0, delta_gs)
    soft_growth = min(growth, GS_SOFT_BUDGET)
    heavy_growth = max(0.0, growth - GS_SOFT_BUDGET)
    return GS_PENALTY_PER_10K * (soft_growth / GS_UNIT) + GS_HEAVY_PENALTY_PER_10K * (heavy_growth / GS_UNIT)


def qcgi(row: dict[str, object], ref: dict[str, object]) -> float:
    return quality_gain(row, ref) - gs_penalty(as_float(row, "gs_num") - as_float(ref, "gs_num"))


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_residual_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "scene": row["scene"],
        "method": "residual_orientation",
        "psnr": row["psnr"],
        "ssim": row["ssim"],
        "lpips": row["lpips"],
        "gs_num": row["gs_num"],
        "train_time_s": row["train_time_s"],
        "run_dir": row["run_dir"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = {row["scene"]: row for row in read_rows(BASELINE)}

    method_rows: dict[str, dict[str, dict[str, object]]] = {}
    for method, path in METHODS.items():
        rows = read_rows(path)
        if method == "residual_orientation":
            rows = [normalize_residual_row(row) for row in rows]
        method_rows[method] = {row["scene"]: row for row in rows}

    comparison_rows: list[dict[str, object]] = []
    for scene in SCENES:
        ref = baseline[scene]
        comparison_rows.append(
            {
                "scene": scene,
                "method": "phase0",
                "delta_psnr": 0.0,
                "delta_ssim": 0.0,
                "delta_lpips": 0.0,
                "delta_gs_num": 0.0,
                "qcgi": 0.0,
            }
        )
        for method, rows_by_scene in method_rows.items():
            row = rows_by_scene[scene]
            comparison_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "delta_psnr": as_float(row, "psnr") - as_float(ref, "psnr"),
                    "delta_ssim": as_float(row, "ssim") - as_float(ref, "ssim"),
                    "delta_lpips": as_float(row, "lpips") - as_float(ref, "lpips"),
                    "delta_gs_num": as_float(row, "gs_num") - as_float(ref, "gs_num"),
                    "qcgi": qcgi(row, ref),
                }
            )

    oracle_rows: list[dict[str, object]] = []
    for scene in SCENES:
        candidates = [row for row in comparison_rows if row["scene"] == scene]
        best = max(candidates, key=lambda row: float(row["qcgi"]))
        oracle_rows.append({"scene": scene, **best})

    methods = ["phase0", *METHODS.keys()]
    policy_rows = []
    for method in methods:
        rows = [row for row in comparison_rows if row["method"] == method]
        policy_rows.append(
            {
                "policy": method,
                "avg_delta_psnr": avg([float(row["delta_psnr"]) for row in rows]),
                "avg_delta_ssim": avg([float(row["delta_ssim"]) for row in rows]),
                "avg_delta_lpips": avg([float(row["delta_lpips"]) for row in rows]),
                "avg_delta_gs_num": avg([float(row["delta_gs_num"]) for row in rows]),
                "avg_qcgi": avg([float(row["qcgi"]) for row in rows]),
            }
        )
    policy_rows.append(
        {
            "policy": "oracle_best_qcgi_per_scene",
            "avg_delta_psnr": avg([float(row["delta_psnr"]) for row in oracle_rows]),
            "avg_delta_ssim": avg([float(row["delta_ssim"]) for row in oracle_rows]),
            "avg_delta_lpips": avg([float(row["delta_lpips"]) for row in oracle_rows]),
            "avg_delta_gs_num": avg([float(row["delta_gs_num"]) for row in oracle_rows]),
            "avg_qcgi": avg([float(row["qcgi"]) for row in oracle_rows]),
        }
    )

    write_csv(
        OUT_DIR / "method_comparison_vs_phase0.csv",
        comparison_rows,
        ["scene", "method", "delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num", "qcgi"],
    )
    write_csv(
        OUT_DIR / "oracle_selection.csv",
        oracle_rows,
        ["scene", "method", "delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num", "qcgi"],
    )
    write_csv(
        OUT_DIR / "policy_summary.csv",
        policy_rows,
        ["policy", "avg_delta_psnr", "avg_delta_ssim", "avg_delta_lpips", "avg_delta_gs_num", "avg_qcgi"],
    )

    summary = {
        "scene_count": len(SCENES),
        "oracle_selection": {row["scene"]: row["method"] for row in oracle_rows},
        "policies": policy_rows,
    }
    (OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
