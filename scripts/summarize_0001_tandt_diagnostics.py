#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips")
METHOD_SPECS = (
    {
        "path_arg": "tandt_summary",
        "methods": {
            "baseline": ("baseline", "control", "FastGS baseline，Tandt 当前回退默认。"),
            "cached_edge_staged142": ("cached_edge_staged142", "candidate", "cached-edge v1，Tandt 负例。"),
        },
    },
    {
        "path_arg": "dino_i050_summary",
        "methods": {
            "dinov2_token_edge_weighted_i050": (
                "dino_weighted_i050",
                "candidate",
                "Tandt 上相对 cached-edge 修复，但低于 baseline。",
            ),
        },
    },
    {
        "path_arg": "dino_i075_summary",
        "methods": {
            "dinov2_token_edge_weighted_i075": (
                "dino_weighted_i075",
                "candidate",
                "高权重 DINO weighted 诊断，Tandt 上低于 i0.50。",
            ),
        },
    },
    {
        "path_arg": "dino_i090_summary",
        "methods": {
            "dinov2_token_edge_weighted_i090": (
                "dino_weighted_i090",
                "candidate",
                "激进权重 DINO weighted 诊断，Tandt 上继续退化。",
            ),
        },
    },
    {
        "path_arg": "dino_i050_auto_prunemin_summary",
        "methods": {
            "dinov2_token_edge_weighted_i050_auto_prunemin": (
                "dino_weighted_i050_auto_prunemin",
                "diagnostic",
                "staged target + 自动容量下限，验证最终补容量不能修复 Tandt。",
            ),
        },
    },
    {
        "path_arg": "dino_i050_prunemin_only_summary",
        "methods": {
            "dinov2_token_edge_weighted_i050_prunemin_only": (
                "dino_weighted_i050_prunemin_only",
                "diagnostic",
                "仅容量下限、不启用 staged target，隔离早期时序影响。",
            ),
        },
    },
    {
        "path_arg": "dino_i050_prune0_summary",
        "methods": {
            "dinov2_token_edge_weighted_i050_prune0": (
                "dino_weighted_i050_prune0",
                "diagnostic",
                "关闭 VFM pruning fusion，隔离 pruning score 融合影响。",
            ),
        },
    },
)


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


def delta(value: float | int | None, reference: float | int | None) -> float | int | None:
    if value is None or reference is None:
        return None
    return value - reference


def metric_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    return row.get("avg_{}".format(key))


def quality_status(candidate: dict[str, Any] | None, reference: dict[str, Any] | None) -> str:
    if candidate is None or reference is None:
        return "missing"
    checks = [
        optional_float(metric_value(candidate, "psnr")) >= optional_float(metric_value(reference, "psnr")),
        optional_float(metric_value(candidate, "ssim")) >= optional_float(metric_value(reference, "ssim")),
        optional_float(metric_value(candidate, "lpips")) <= optional_float(metric_value(reference, "lpips")),
    ]
    if all(checks):
        return "all3_positive"
    if sum(1 for item in checks if item) >= 2:
        return "mixed_positive"
    return "below_reference"


def psnr_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        optional_float(metric_value(row, "psnr")) or float("-inf"),
        -(optional_float(metric_value(row, "lpips")) or float("inf")),
        optional_float(metric_value(row, "ssim")) or float("-inf"),
    )


def lpips_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -(optional_float(metric_value(row, "lpips")) or float("inf")),
        optional_float(metric_value(row, "psnr")) or float("-inf"),
        optional_float(metric_value(row, "ssim")) or float("-inf"),
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        path = getattr(args, spec["path_arg"])
        if not path.exists():
            continue
        for item in read_csv(path):
            raw_method = str(item.get("method", ""))
            if raw_method not in spec["methods"]:
                continue
            method, category, note = spec["methods"][raw_method]
            rows.append(
                {
                    "dataset": "tandt",
                    "scene": str(item["scene"]),
                    "method": method,
                    "category": category,
                    "target_gaussian_count": optional_int(item.get("target_gaussian_count")),
                    "psnr": optional_float(item.get("psnr")),
                    "ssim": optional_float(item.get("ssim")),
                    "lpips": optional_float(item.get("lpips")),
                    "gs_num": optional_int(item.get("gs_num")),
                    "train_time_s": optional_float(item.get("train_time_s")),
                    "run_dir": str(item.get("run_dir", "")),
                    "source_summary": str(path),
                    "note": note,
                }
            )
    order = {name: index for index, name in enumerate(
        [
            "baseline",
            "cached_edge_staged142",
            "dino_weighted_i050",
            "dino_weighted_i075",
            "dino_weighted_i090",
            "dino_weighted_i050_auto_prunemin",
            "dino_weighted_i050_prunemin_only",
            "dino_weighted_i050_prune0",
        ]
    )}
    return sorted(rows, key=lambda row: (str(row["scene"]), order.get(str(row["method"]), 99)))


def by_scene_method(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["scene"]))][str(row["method"])] = row
    return grouped


def build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for (dataset, scene), methods in sorted(by_scene_method(rows).items()):
        for method, row in sorted(methods.items()):
            for ref_method in ("baseline", "cached_edge_staged142", "dino_weighted_i050"):
                if method == ref_method or ref_method not in methods:
                    continue
                ref = methods[ref_method]
                comparisons.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "method": method,
                        "reference": ref_method,
                        "delta_psnr": delta(optional_float(row["psnr"]), optional_float(ref["psnr"])),
                        "delta_ssim": delta(optional_float(row["ssim"]), optional_float(ref["ssim"])),
                        "delta_lpips": delta(optional_float(row["lpips"]), optional_float(ref["lpips"])),
                        "delta_gs_num": delta(optional_int(row["gs_num"]), optional_int(ref["gs_num"])),
                        "delta_train_time_s": delta(
                            optional_float(row["train_time_s"]), optional_float(ref["train_time_s"])
                        ),
                        "quality_status": quality_status(row, ref),
                    }
                )
    return comparisons


def build_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    averages: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        avg: dict[str, Any] = {
            "dataset": "tandt",
            "method": method,
            "category": items[0]["category"],
            "scene_count": len(items),
        }
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            values = [float(row[key]) for row in items if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = mean(values)
        averages.append(avg)
    return averages


def build_policy(rows: list[dict[str, Any]], averages: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = by_scene_method(rows)
    scene_rows = []
    for (dataset, scene), methods in sorted(grouped.items()):
        ordered = list(methods.values())
        baseline = methods.get("baseline")
        best_psnr = max(ordered, key=psnr_key)
        best_lpips = max(ordered, key=lpips_key)
        positive_vs_baseline = [
            row for row in ordered
            if row["method"] != "baseline" and quality_status(row, baseline) == "all3_positive"
        ]
        if baseline is not None and not positive_vs_baseline:
            pick = baseline
            reason = "没有候选三项指标同时超过 baseline，回退 baseline。"
        else:
            pick = max(positive_vs_baseline, key=psnr_key) if positive_vs_baseline else best_psnr
            reason = "选择三项指标同时超过 baseline 的最高 PSNR 候选。"
        scene_rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "policy_pick": pick["method"],
                "policy_reason": reason,
                "best_psnr_method": best_psnr["method"],
                "best_psnr": best_psnr["psnr"],
                "best_lpips_method": best_lpips["method"],
                "best_lpips": best_lpips["lpips"],
            }
        )

    avg_by_method = {str(row["method"]): row for row in averages}
    baseline_avg = avg_by_method.get("baseline")
    positive_avg_methods = [
        row for row in averages
        if row["method"] != "baseline" and quality_status(row, baseline_avg) == "all3_positive"
    ]
    if baseline_avg is not None and not positive_avg_methods:
        dataset_pick = "baseline"
        dataset_reason = "Tandt 所有 VFM 候选/诊断的平均指标都没有三项同时超过 baseline。"
    else:
        pick = max(positive_avg_methods, key=psnr_key) if positive_avg_methods else max(averages, key=psnr_key)
        dataset_pick = pick["method"]
        dataset_reason = "存在平均三项同时超过 baseline 的候选。"

    return {
        "dataset": "tandt",
        "dataset_policy_pick": dataset_pick,
        "dataset_policy_reason": dataset_reason,
        "scene_policies": scene_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Tandt diagnostics for experiment 0001.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/0001/tandt_diagnostics"))
    parser.add_argument("--tandt-summary", type=Path, default=Path("output/0001/full_tandt_db_v1/tandt/summary.csv"))
    parser.add_argument("--dino-i050-summary", type=Path, default=Path("output/0001/dino_weighted_i050_tandt/summary.csv"))
    parser.add_argument("--dino-i075-summary", type=Path, default=Path("output/0001/dino_weighted_i075_tandt/summary.csv"))
    parser.add_argument("--dino-i090-summary", type=Path, default=Path("output/0001/dino_weighted_i090_tandt/summary.csv"))
    parser.add_argument(
        "--dino-i050-auto-prunemin-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i050_auto_prunemin_tandt/summary.csv"),
    )
    parser.add_argument(
        "--dino-i050-prunemin-only-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i050_prunemin_only_tandt/summary.csv"),
    )
    parser.add_argument(
        "--dino-i050-prune0-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i050_prune0_tandt/summary.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    comparisons = build_comparisons(summary)
    averages = build_averages(summary)
    policy = build_policy(summary, averages)

    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    write_json(args.output_dir / "comparisons.json", comparisons)
    write_csv(args.output_dir / "averages.csv", averages)
    write_json(args.output_dir / "averages.json", averages)
    write_csv(args.output_dir / "scene_policy.csv", policy["scene_policies"])
    write_json(args.output_dir / "policy.json", policy)

    print("Wrote {}".format(args.output_dir / "summary.csv"))
    print("Wrote {}".format(args.output_dir / "policy.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
