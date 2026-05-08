#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_MAP = {
    "baseline": "baseline",
    "cached_edge_staged142": "cached_edge_staged142",
    "weighted_i050": "dino_weighted_i050",
    "dinov2_token_edge_weighted_i050": "dino_weighted_i050",
}
METHOD_ORDER = {
    "baseline": 0,
    "cached_edge_staged142": 1,
    "dino_weighted_i050": 2,
}
METRIC_KEYS = ("psnr", "ssim", "lpips")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def append_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    path: Path,
    allowed_methods: set[str],
    source: str,
) -> None:
    for item in read_csv(path):
        raw_method = str(item.get("method", ""))
        if raw_method not in allowed_methods:
            continue
        method = METHOD_MAP.get(raw_method, raw_method)
        rows.append(
            {
                "dataset": dataset,
                "scene": str(item["scene"]),
                "method": method,
                "psnr": optional_float(item.get("psnr")),
                "ssim": optional_float(item.get("ssim")),
                "lpips": optional_float(item.get("lpips")),
                "gs_num": optional_int(item.get("gs_num")),
                "train_time_s": optional_float(item.get("train_time_s")),
                "run_dir": str(item.get("run_dir", "")),
                "source_summary": source,
            }
        )


def method_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    method = str(row["method"])
    return METHOD_ORDER.get(method, 99), method


def scene_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["scene"])


def build_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    append_rows(
        rows,
        dataset="mipnerf360",
        path=args.mipnerf_summary,
        allowed_methods={"baseline", "cached_edge_staged142"},
        source=str(args.mipnerf_summary),
    )
    append_rows(
        rows,
        dataset="mipnerf360",
        path=args.mipnerf_weighted_summary,
        allowed_methods={"weighted_i050"},
        source=str(args.mipnerf_weighted_summary),
    )
    append_rows(
        rows,
        dataset="tandt",
        path=args.tandt_summary,
        allowed_methods={"baseline", "cached_edge_staged142"},
        source=str(args.tandt_summary),
    )
    append_rows(
        rows,
        dataset="tandt",
        path=args.tandt_weighted_summary,
        allowed_methods={"dinov2_token_edge_weighted_i050"},
        source=str(args.tandt_weighted_summary),
    )
    append_rows(
        rows,
        dataset="db",
        path=args.db_summary,
        allowed_methods={"baseline", "cached_edge_staged142"},
        source=str(args.db_summary),
    )
    append_rows(
        rows,
        dataset="db",
        path=args.db_weighted_summary,
        allowed_methods={"dinov2_token_edge_weighted_i050"},
        source=str(args.db_weighted_summary),
    )
    return sorted(rows, key=lambda row: (str(row["dataset"]), str(row["scene"]), method_sort_key(row)))


def build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[scene_key(row)][str(row["method"])] = row

    comparisons: list[dict[str, Any]] = []
    for (dataset, scene), methods in sorted(by_scene.items()):
        for method, row in sorted(methods.items(), key=lambda item: METHOD_ORDER.get(item[0], 99)):
            for reference_name in ("baseline", "cached_edge_staged142"):
                if method == reference_name or reference_name not in methods:
                    continue
                reference = methods[reference_name]
                comparisons.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "method": method,
                        "reference": reference_name,
                        "delta_psnr": delta(optional_float(row["psnr"]), optional_float(reference["psnr"])),
                        "delta_ssim": delta(optional_float(row["ssim"]), optional_float(reference["ssim"])),
                        "delta_lpips": delta(optional_float(row["lpips"]), optional_float(reference["lpips"])),
                        "delta_gs_num": delta(optional_int(row["gs_num"]), optional_int(reference["gs_num"])),
                        "delta_train_time_s": delta(
                            optional_float(row["train_time_s"]), optional_float(reference["train_time_s"])
                        ),
                    }
                )
    return comparisons


def no_worse(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return (
        optional_float(candidate["psnr"]) is not None
        and optional_float(reference["psnr"]) is not None
        and optional_float(candidate["psnr"]) >= optional_float(reference["psnr"])
        and optional_float(candidate["ssim"]) >= optional_float(reference["ssim"])
        and optional_float(candidate["lpips"]) <= optional_float(reference["lpips"])
    )


def quality_status(candidate: dict[str, Any] | None, reference: dict[str, Any] | None) -> str:
    if candidate is None or reference is None:
        return "missing"
    checks = [
        optional_float(candidate["psnr"]) >= optional_float(reference["psnr"]),
        optional_float(candidate["ssim"]) >= optional_float(reference["ssim"]),
        optional_float(candidate["lpips"]) <= optional_float(reference["lpips"]),
    ]
    if all(checks):
        return "all3_positive"
    if sum(1 for item in checks if item) >= 2:
        return "mixed_positive"
    return "below_reference"


def psnr_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        optional_float(row["psnr"]) or float("-inf"),
        -(optional_float(row["lpips"]) or float("inf")),
        optional_float(row["ssim"]) or float("-inf"),
    )


def lpips_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -(optional_float(row["lpips"]) or float("inf")),
        optional_float(row["psnr"]) or float("-inf"),
        optional_float(row["ssim"]) or float("-inf"),
    )


def build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[scene_key(row)][str(row["method"])] = row

    recommendations: list[dict[str, Any]] = []
    for (dataset, scene), methods in sorted(by_scene.items()):
        ordered = list(methods.values())
        baseline = methods.get("baseline")
        cached = methods.get("cached_edge_staged142")
        dino = methods.get("dino_weighted_i050")
        best_psnr = max(ordered, key=psnr_key)
        best_lpips = max(ordered, key=lpips_key)

        budget_candidates = ordered
        if baseline is not None:
            budget_candidates = [row for row in ordered if no_worse(row, baseline)]
        budget_pick = min(
            budget_candidates or [baseline or best_psnr],
            key=lambda row: (optional_int(row["gs_num"]) if optional_int(row["gs_num"]) is not None else 10**18, -psnr_key(row)[0]),
        )

        vfm_candidates = [row for row in ordered if row["method"] != "baseline"]
        vfm_pick = max(vfm_candidates, key=psnr_key) if vfm_candidates else best_psnr

        if best_psnr["method"] == "baseline":
            reason = "baseline 的 PSNR 最高，当前场景不应默认启用 VFM 后端。"
        elif best_psnr["method"] == "cached_edge_staged142":
            reason = "cached-edge 是该场景 PSNR 最优的 VFM proxy。"
        else:
            reason = "DINO weighted i0.50 是该场景 PSNR 最优的 VFM 候选。"

        recommendations.append(
            {
                "dataset": dataset,
                "scene": scene,
                "best_psnr_method": best_psnr["method"],
                "best_psnr": best_psnr["psnr"],
                "best_lpips_method": best_lpips["method"],
                "best_lpips": best_lpips["lpips"],
                "budget_no_worse_method": budget_pick["method"],
                "budget_no_worse_gs_num": budget_pick["gs_num"],
                "vfm_psnr_pick": vfm_pick["method"],
                "cached_vs_baseline_status": quality_status(cached, baseline),
                "dino_weighted_vs_baseline_status": quality_status(dino, baseline),
                "dino_weighted_vs_cached_status": quality_status(dino, cached),
                "reason": reason,
            }
        )
    return recommendations


def lookup_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["scene"]), str(row["method"])): row
        for row in rows
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_recommendation_averages(
    rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = lookup_rows(rows)
    pick_columns = (
        ("best_psnr_method", "best_psnr_oracle"),
        ("best_lpips_method", "best_lpips_oracle"),
        ("budget_no_worse_method", "budget_no_worse"),
        ("vfm_psnr_pick", "vfm_psnr_pick"),
    )
    averages: list[dict[str, Any]] = []
    for method_column, label in pick_columns:
        picked_rows = []
        for recommendation in recommendations:
            key = (
                str(recommendation["dataset"]),
                str(recommendation["scene"]),
                str(recommendation[method_column]),
            )
            row = by_key.get(key)
            if row is not None:
                picked_rows.append(row)
        avg: dict[str, Any] = {"selector": label, "scene_count": len(picked_rows)}
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            values = [float(row[key]) for row in picked_rows if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = mean(values)
        averages.append(avg)
    return averages


def build_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)
        grouped[("all", str(row["method"]))].append(row)

    averages: list[dict[str, Any]] = []
    for (dataset, method), items in sorted(grouped.items()):
        avg: dict[str, Any] = {"dataset": dataset, "method": method, "scene_count": len(items)}
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            values = [float(row[key]) for row in items if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = mean(values)
        averages.append(avg)
    return averages


def build_average_comparisons(averages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in averages:
        by_dataset[str(row["dataset"])][str(row["method"])] = row

    comparisons: list[dict[str, Any]] = []
    for dataset, methods in sorted(by_dataset.items()):
        for method, row in sorted(methods.items(), key=lambda item: METHOD_ORDER.get(item[0], 99)):
            for reference_name in ("baseline", "cached_edge_staged142"):
                if method == reference_name or reference_name not in methods:
                    continue
                reference = methods[reference_name]
                comparisons.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reference": reference_name,
                        "delta_avg_psnr": delta(row.get("avg_psnr"), reference.get("avg_psnr")),
                        "delta_avg_ssim": delta(row.get("avg_ssim"), reference.get("avg_ssim")),
                        "delta_avg_lpips": delta(row.get("avg_lpips"), reference.get("avg_lpips")),
                        "delta_avg_gs_num": delta(row.get("avg_gs_num"), reference.get("avg_gs_num")),
                        "delta_avg_train_time_s": delta(
                            row.get("avg_train_time_s"), reference.get("avg_train_time_s")
                        ),
                    }
                )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 0001 backend selection across datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/0001/cross_dataset_selector"))
    parser.add_argument("--mipnerf-summary", type=Path, default=Path("output/0001/full_mipnerf360_v1/summary.csv"))
    parser.add_argument(
        "--mipnerf-weighted-summary",
        type=Path,
        default=Path("output/0001/weighted_candidate_summary/summary.csv"),
    )
    parser.add_argument("--tandt-summary", type=Path, default=Path("output/0001/full_tandt_db_v1/tandt/summary.csv"))
    parser.add_argument(
        "--tandt-weighted-summary", type=Path, default=Path("output/0001/dino_weighted_i050_tandt/summary.csv")
    )
    parser.add_argument("--db-summary", type=Path, default=Path("output/0001/full_tandt_db_v1/db/summary.csv"))
    parser.add_argument(
        "--db-weighted-summary", type=Path, default=Path("output/0001/dino_weighted_i050_db/summary.csv")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    comparisons = build_comparisons(summary)
    recommendations = build_recommendations(summary)
    averages = build_averages(summary)
    average_comparisons = build_average_comparisons(averages)
    recommendation_averages = build_recommendation_averages(summary, recommendations)

    outputs = {
        "summary": summary,
        "comparisons": comparisons,
        "recommendations": recommendations,
        "averages": averages,
        "average_comparisons": average_comparisons,
        "recommendation_averages": recommendation_averages,
    }
    for name, rows in outputs.items():
        write_csv(output_dir / "{}.csv".format(name), rows)
        write_json(output_dir / "{}.json".format(name), rows)

    print("Wrote {}".format(output_dir / "summary.csv"))
    print("Wrote {}".format(output_dir / "recommendations.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
