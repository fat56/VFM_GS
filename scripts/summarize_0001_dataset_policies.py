#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips", "gs_num", "train_time_s")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def metric_value(row: dict[str, Any], key: str) -> float | int | None:
    if key in ("gs_num",):
        return optional_int(row.get(key))
    return optional_float(row.get(key))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def lookup_summary(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = {}
    for item in read_csv(path):
        row = {
            "dataset": item["dataset"],
            "scene": item["scene"],
            "method": item["method"],
            "psnr": optional_float(item.get("psnr")),
            "ssim": optional_float(item.get("ssim")),
            "lpips": optional_float(item.get("lpips")),
            "gs_num": optional_int(item.get("gs_num")),
            "train_time_s": optional_float(item.get("train_time_s")),
            "run_dir": item.get("run_dir", ""),
        }
        rows[(row["dataset"], row["scene"], row["method"])] = row
    return rows


def lookup_weighted_recommendations(path: Path) -> dict[str, dict[str, str]]:
    return {row["scene"]: row for row in read_csv(path)}


def method_for_policy(
    policy: str,
    dataset: str,
    scene: str,
    weighted_recommendations: dict[str, dict[str, str]],
) -> str:
    if policy == "baseline":
        return "baseline"
    if policy == "dataset_fixed_policy":
        if dataset == "mipnerf360":
            return "dino_weighted_i050"
        if dataset == "db":
            return "dino_weighted_i090"
        return "baseline"
    if policy == "dataset_quality_policy":
        if dataset == "mipnerf360":
            pick = weighted_recommendations[scene]["qcgi_pick"]
            return {
                "weighted_i050": "dino_weighted_i050",
                "weighted_i075": "dino_weighted_i075",
                "weighted_i090": "dino_weighted_i090",
            }[pick]
        if dataset == "db":
            return "dino_weighted_i090"
        return "baseline"
    raise ValueError("Unknown policy: {}".format(policy))


def build_policy_rows(
    summary: dict[tuple[str, str, str], dict[str, Any]],
    weighted_recommendations: dict[str, dict[str, str]],
    policies: list[str],
) -> list[dict[str, Any]]:
    scenes = sorted({(dataset, scene) for dataset, scene, _ in summary})
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for dataset, scene in scenes:
            method = method_for_policy(policy, dataset, scene, weighted_recommendations)
            picked = summary[(dataset, scene, method)]
            rows.append(
                {
                    "policy": policy,
                    "dataset": dataset,
                    "scene": scene,
                    "method": method,
                    "psnr": picked["psnr"],
                    "ssim": picked["ssim"],
                    "lpips": picked["lpips"],
                    "gs_num": picked["gs_num"],
                    "train_time_s": picked["train_time_s"],
                    "run_dir": picked["run_dir"],
                    "reason": policy_reason(policy, dataset),
                }
            )
    return rows


def policy_reason(policy: str, dataset: str) -> str:
    if policy == "baseline":
        return "统一 baseline 控制线。"
    if policy == "dataset_fixed_policy":
        if dataset == "mipnerf360":
            return "MipNeRF360 固定采用 weighted i0.50 预算效率档。"
        if dataset == "db":
            return "DB 固定采用 DINO weighted i0.90 高质量档。"
        return "Tandt 诊断表要求 baseline 回退。"
    if policy == "dataset_quality_policy":
        if dataset == "mipnerf360":
            return "MipNeRF360 采用 weighted QCGI 场景选择。"
        if dataset == "db":
            return "DB 固定采用 DINO weighted i0.90 高质量档。"
        return "Tandt 诊断表要求 baseline 回退。"
    return ""


def build_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], "all")].append(row)
        grouped[(row["policy"], row["dataset"])].append(row)

    averages: list[dict[str, Any]] = []
    for (policy, dataset), items in sorted(grouped.items()):
        avg: dict[str, Any] = {"policy": policy, "dataset": dataset, "scene_count": len(items)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in items if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = mean(values)
        averages.append(avg)
    return averages


def build_comparisons(averages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy_dataset = {
        (row["policy"], row["dataset"]): row
        for row in averages
    }
    comparisons: list[dict[str, Any]] = []
    for row in averages:
        if row["policy"] == "baseline":
            continue
        ref = by_policy_dataset.get(("baseline", row["dataset"]))
        if ref is None:
            continue
        comparison = {
            "policy": row["policy"],
            "dataset": row["dataset"],
            "reference": "baseline",
            "scene_count": row["scene_count"],
        }
        for key in METRIC_KEYS:
            comparison["delta_avg_{}".format(key)] = (
                None
                if row.get("avg_{}".format(key)) is None or ref.get("avg_{}".format(key)) is None
                else row["avg_{}".format(key)] - ref["avg_{}".format(key)]
            )
        comparisons.append(comparison)
    return comparisons


def dataset_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("dataset") != "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize preset dataset policies for experiment 0001.")
    parser.add_argument("--summary", type=Path, default=Path("output/0001/cross_dataset_selector/summary.csv"))
    parser.add_argument(
        "--weighted-recommendations",
        type=Path,
        default=Path("output/0001/weighted_candidate_summary/recommendations.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/0001/dataset_policies"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = lookup_summary(args.summary)
    weighted_recommendations = lookup_weighted_recommendations(args.weighted_recommendations)
    policies = ["baseline", "dataset_fixed_policy", "dataset_quality_policy"]
    policy_rows = build_policy_rows(summary, weighted_recommendations, policies)
    averages = build_averages(policy_rows)
    comparisons = build_comparisons(averages)
    dataset_averages = dataset_only(averages)
    dataset_comparisons = dataset_only(comparisons)

    write_csv(args.output_dir / "policy_rows.csv", policy_rows)
    write_json(args.output_dir / "policy_rows.json", policy_rows)
    write_csv(args.output_dir / "averages.csv", averages)
    write_json(args.output_dir / "averages.json", averages)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    write_json(args.output_dir / "comparisons.json", comparisons)
    write_csv(args.output_dir / "dataset_averages.csv", dataset_averages)
    write_json(args.output_dir / "dataset_averages.json", dataset_averages)
    write_csv(args.output_dir / "dataset_comparisons.csv", dataset_comparisons)
    write_json(args.output_dir / "dataset_comparisons.json", dataset_comparisons)

    print("Wrote {}".format(args.output_dir / "averages.csv"))
    print("Wrote {}".format(args.output_dir / "comparisons.csv"))
    print("Wrote {}".format(args.output_dir / "dataset_averages.csv"))
    print("Wrote {}".format(args.output_dir / "dataset_comparisons.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
