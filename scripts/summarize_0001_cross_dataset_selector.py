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
    "weighted_i075": "dino_weighted_i075",
    "dinov2_token_edge_weighted_i075": "dino_weighted_i075",
    "weighted_i090": "dino_weighted_i090",
    "dinov2_token_edge_weighted_i090": "dino_weighted_i090",
}
METHOD_ORDER = {
    "baseline": 0,
    "cached_edge_staged142": 1,
    "dino_weighted_i050": 2,
    "dino_weighted_i075": 3,
    "dino_weighted_i090": 4,
}
METRIC_KEYS = ("psnr", "ssim", "lpips")
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04
QUALITY_WEIGHTS = {
    "psnr": 1.0,
    "ssim": 20.0,
    "lpips": 5.0,
}


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


def quality_gain(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    d_psnr = delta(optional_float(candidate["psnr"]), optional_float(reference["psnr"]))
    d_ssim = delta(optional_float(candidate["ssim"]), optional_float(reference["ssim"]))
    d_lpips = delta(optional_float(candidate["lpips"]), optional_float(reference["lpips"]))
    if d_psnr is None or d_ssim is None or d_lpips is None:
        return None
    return (
        QUALITY_WEIGHTS["psnr"] * d_psnr
        + QUALITY_WEIGHTS["ssim"] * d_ssim
        - QUALITY_WEIGHTS["lpips"] * d_lpips
    )


def gs_growth_band(delta_gs_num: int | float | None) -> str:
    if delta_gs_num is None:
        return "unknown"
    if delta_gs_num <= 0:
        return "no_growth"
    if delta_gs_num < GS_UNIT:
        return "sub_0.01M"
    if delta_gs_num < GS_SOFT_BUDGET:
        return "0.01M_to_0.10M"
    return "gte_0.10M"


def gs_penalty(delta_gs_num: int | float | None) -> float | None:
    if delta_gs_num is None:
        return None
    growth = max(0.0, float(delta_gs_num))
    soft_growth = min(growth, GS_SOFT_BUDGET)
    heavy_growth = max(0.0, growth - GS_SOFT_BUDGET)
    return (
        GS_PENALTY_PER_10K * (soft_growth / GS_UNIT)
        + GS_HEAVY_PENALTY_PER_10K * (heavy_growth / GS_UNIT)
    )


def qcgi(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    gain = quality_gain(candidate, reference)
    penalty = gs_penalty(delta(optional_int(candidate["gs_num"]), optional_int(reference["gs_num"])))
    if gain is None or penalty is None:
        return None
    return gain - penalty


def quality_per_10k(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    gain = quality_gain(candidate, reference)
    d_gs = delta(optional_int(candidate["gs_num"]), optional_int(reference["gs_num"]))
    if gain is None or d_gs is None:
        return None
    if d_gs <= 0:
        return gain
    return gain / (d_gs / GS_UNIT)


def append_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    path: Path,
    allowed_methods: set[str],
    source: str,
    missing_ok: bool = False,
) -> None:
    if missing_ok and not path.exists():
        return
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
        allowed_methods={"weighted_i050", "weighted_i075", "weighted_i090"},
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
        dataset="tandt",
        path=args.tandt_weighted_i075_summary,
        allowed_methods={"dinov2_token_edge_weighted_i075"},
        source=str(args.tandt_weighted_i075_summary),
        missing_ok=True,
    )
    append_rows(
        rows,
        dataset="tandt",
        path=args.tandt_weighted_i090_summary,
        allowed_methods={"dinov2_token_edge_weighted_i090"},
        source=str(args.tandt_weighted_i090_summary),
        missing_ok=True,
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
    append_rows(
        rows,
        dataset="db",
        path=args.db_weighted_i075_summary,
        allowed_methods={"dinov2_token_edge_weighted_i075"},
        source=str(args.db_weighted_i075_summary),
        missing_ok=True,
    )
    append_rows(
        rows,
        dataset="db",
        path=args.db_weighted_i090_summary,
        allowed_methods={"dinov2_token_edge_weighted_i090"},
        source=str(args.db_weighted_i090_summary),
        missing_ok=True,
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
                        "quality_gain": quality_gain(row, reference),
                        "quality_gain_per_10k_gs": quality_per_10k(row, reference),
                        "gs_growth_band": gs_growth_band(
                            delta(optional_int(row["gs_num"]), optional_int(reference["gs_num"]))
                        ),
                        "gs_penalty": gs_penalty(
                            delta(optional_int(row["gs_num"]), optional_int(reference["gs_num"]))
                        ),
                        "qcgi": qcgi(row, reference),
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


def select_validated_policy(
    baseline: dict[str, Any] | None,
    cached: dict[str, Any] | None,
    dino_candidates: list[dict[str, Any]],
    best_psnr: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    cached_vs_baseline = quality_status(cached, baseline)
    dino_vs_baseline_candidates = [
        row for row in dino_candidates if quality_status(row, baseline) == "all3_positive"
    ]
    dino_vs_both_candidates = [
        row for row in dino_vs_baseline_candidates if quality_status(row, cached) == "all3_positive"
    ]

    if dino_vs_both_candidates:
        return max(dino_vs_both_candidates, key=psnr_key), "dino_all3_vs_baseline_and_cached"
    if cached is not None and cached_vs_baseline == "all3_positive":
        return cached, "cached_all3_vs_baseline"
    if dino_vs_baseline_candidates:
        return max(dino_vs_baseline_candidates, key=psnr_key), "dino_all3_vs_baseline"
    if baseline is not None:
        return baseline, "baseline_fallback"
    return best_psnr, "best_psnr_fallback"


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


def qcgi_pick_key(row: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[float, float, float]:
    if baseline is None or row["method"] == "baseline":
        score = 0.0
    else:
        score = qcgi(row, baseline)
        if score is None:
            score = float("-inf")
    return (score, optional_float(row["psnr"]) or float("-inf"), -(optional_float(row["lpips"]) or float("inf")))


def build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[scene_key(row)][str(row["method"])] = row

    recommendations: list[dict[str, Any]] = []
    for (dataset, scene), methods in sorted(by_scene.items()):
        ordered = list(methods.values())
        baseline = methods.get("baseline")
        cached = methods.get("cached_edge_staged142")
        dino_candidates = [
            methods[name]
            for name in ("dino_weighted_i050", "dino_weighted_i075", "dino_weighted_i090")
            if name in methods
        ]
        best_dino = max(dino_candidates, key=psnr_key) if dino_candidates else None
        best_psnr = max(ordered, key=psnr_key)
        best_lpips = max(ordered, key=lpips_key)
        qcgi_pick = max(ordered, key=lambda row: qcgi_pick_key(row, baseline))
        qcgi_pick_score = 0.0 if qcgi_pick["method"] == "baseline" or baseline is None else qcgi(qcgi_pick, baseline)
        validated_policy, validated_policy_reason = select_validated_policy(
            baseline, cached, dino_candidates, best_psnr
        )

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
        elif str(best_psnr["method"]).startswith("dino_weighted_"):
            reason = "{} 是该场景 PSNR 最优的 DINO weighted 候选。".format(best_psnr["method"])
        else:
            reason = "{} 是该场景 PSNR 最优的 VFM 候选。".format(best_psnr["method"])

        recommendations.append(
            {
                "dataset": dataset,
                "scene": scene,
                "best_psnr_method": best_psnr["method"],
                "best_psnr": best_psnr["psnr"],
                "best_lpips_method": best_lpips["method"],
                "best_lpips": best_lpips["lpips"],
                "best_dino_method": best_dino["method"] if best_dino is not None else "",
                "best_dino_psnr": best_dino["psnr"] if best_dino is not None else "",
                "qcgi_pick_method": qcgi_pick["method"],
                "qcgi_pick_score": qcgi_pick_score,
                "validated_policy_method": validated_policy["method"],
                "validated_policy_reason": validated_policy_reason,
                "budget_no_worse_method": budget_pick["method"],
                "budget_no_worse_gs_num": budget_pick["gs_num"],
                "vfm_psnr_pick": vfm_pick["method"],
                "cached_vs_baseline_status": quality_status(cached, baseline),
                "dino_weighted_vs_baseline_status": quality_status(methods.get("dino_weighted_i050"), baseline),
                "dino_weighted_vs_cached_status": quality_status(methods.get("dino_weighted_i050"), cached),
                "best_dino_vs_baseline_status": quality_status(best_dino, baseline),
                "best_dino_vs_cached_status": quality_status(best_dino, cached),
                "dino_weighted_i075_vs_baseline_status": quality_status(methods.get("dino_weighted_i075"), baseline),
                "dino_weighted_i090_vs_baseline_status": quality_status(methods.get("dino_weighted_i090"), baseline),
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
        ("qcgi_pick_method", "qcgi_pick"),
        ("validated_policy_method", "validated_policy"),
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
                        "avg_quality_gain": quality_gain(
                            {
                                "psnr": row.get("avg_psnr"),
                                "ssim": row.get("avg_ssim"),
                                "lpips": row.get("avg_lpips"),
                            },
                            {
                                "psnr": reference.get("avg_psnr"),
                                "ssim": reference.get("avg_ssim"),
                                "lpips": reference.get("avg_lpips"),
                            },
                        ),
                        "avg_quality_gain_per_10k_gs": quality_per_10k(
                            {
                                "psnr": row.get("avg_psnr"),
                                "ssim": row.get("avg_ssim"),
                                "lpips": row.get("avg_lpips"),
                                "gs_num": row.get("avg_gs_num"),
                            },
                            {
                                "psnr": reference.get("avg_psnr"),
                                "ssim": reference.get("avg_ssim"),
                                "lpips": reference.get("avg_lpips"),
                                "gs_num": reference.get("avg_gs_num"),
                            },
                        ),
                        "avg_gs_growth_band": gs_growth_band(
                            delta(row.get("avg_gs_num"), reference.get("avg_gs_num"))
                        ),
                        "avg_gs_penalty": gs_penalty(
                            delta(row.get("avg_gs_num"), reference.get("avg_gs_num"))
                        ),
                        "avg_qcgi": qcgi(
                            {
                                "psnr": row.get("avg_psnr"),
                                "ssim": row.get("avg_ssim"),
                                "lpips": row.get("avg_lpips"),
                                "gs_num": row.get("avg_gs_num"),
                            },
                            {
                                "psnr": reference.get("avg_psnr"),
                                "ssim": reference.get("avg_ssim"),
                                "lpips": reference.get("avg_lpips"),
                                "gs_num": reference.get("avg_gs_num"),
                            },
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
    parser.add_argument(
        "--tandt-weighted-i075-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i075_tandt/summary.csv"),
    )
    parser.add_argument(
        "--tandt-weighted-i090-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i090_tandt/summary.csv"),
    )
    parser.add_argument("--db-summary", type=Path, default=Path("output/0001/full_tandt_db_v1/db/summary.csv"))
    parser.add_argument(
        "--db-weighted-summary", type=Path, default=Path("output/0001/dino_weighted_i050_db/summary.csv")
    )
    parser.add_argument(
        "--db-weighted-i075-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i075_db/summary.csv"),
    )
    parser.add_argument(
        "--db-weighted-i090-summary",
        type=Path,
        default=Path("output/0001/dino_weighted_i090_db/summary.csv"),
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
