#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = ("psnr", "ssim", "lpips")
QUALITY_CANDIDATE_METHODS = ("weighted_i075", "weighted_i090")
QUALITY_MIN_DELTA_PSNR = 0.0
QUALITY_MIN_DELTA_SSIM = -0.0002
QUALITY_MAX_DELTA_LPIPS = 0.0002
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04
QUALITY_WEIGHTS = {
    "psnr": 1.0,
    "ssim": 20.0,
    "lpips": 5.0,
}
GS_RE = re.compile(r"Gaussian number:\s*(\d+)")
TIME_RE = re.compile(r"Training time:\s*([0-9.]+)")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read {}".format(path)) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Catalog must be a YAML mapping: {}".format(path))
    return data


def read_metrics(run_dir: Path) -> dict[str, float | None]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {key: None for key in METRIC_KEYS}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        return {key: None for key in METRIC_KEYS}
    method = sorted(data)[-1]
    values = data.get(method, {})
    return {
        "psnr": values.get("PSNR"),
        "ssim": values.get("SSIM"),
        "lpips": values.get("LPIPS"),
    }


def latest_point_count(run_dir: Path) -> int | None:
    point_cloud_dir = run_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None
    candidates = []
    for child in point_cloud_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            candidates.append((int(child.name.split("_", 1)[1]), child / "point_cloud.ply"))
        except ValueError:
            continue
    for _, ply_path in sorted(candidates, reverse=True):
        count = read_ply_vertex_count(ply_path)
        if count is not None:
            return count
    return None


def read_ply_vertex_count(ply_path: Path) -> int | None:
    if not ply_path.exists():
        return None
    with ply_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                return None
    return None


def parse_train_log(log_path: Path) -> tuple[int | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gs_matches = GS_RE.findall(text)
    time_matches = TIME_RE.findall(text)
    gs_num = int(gs_matches[-1]) if gs_matches else None
    train_time = float(time_matches[-1]) if time_matches else None
    return gs_num, train_time


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def delta(value: float | int | None, reference: float | int | None) -> float | int | None:
    if value is None or reference is None:
        return None
    return value - reference


def quality_gain_from_deltas(deltas: dict[str, float | int | None]) -> float | None:
    d_psnr = deltas.get("delta_psnr")
    d_ssim = deltas.get("delta_ssim")
    d_lpips = deltas.get("delta_lpips")
    if d_psnr is None or d_ssim is None or d_lpips is None:
        return None
    return (
        QUALITY_WEIGHTS["psnr"] * float(d_psnr)
        + QUALITY_WEIGHTS["ssim"] * float(d_ssim)
        - QUALITY_WEIGHTS["lpips"] * float(d_lpips)
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


def qcgi_from_deltas(deltas: dict[str, float | int | None]) -> float | None:
    gain = quality_gain_from_deltas(deltas)
    penalty = gs_penalty(deltas.get("delta_gs"))
    if gain is None or penalty is None:
        return None
    return gain - penalty


def collect_rows(catalog: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    rows = []
    for item in catalog.get("runs", []):
        if not isinstance(item, dict):
            raise ValueError("Each catalog run must be a mapping")
        run_dir = Path(str(item["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = repo / run_dir
        metrics = read_metrics(run_dir)

        log_path = item.get("train_log")
        log_gs = None
        log_train_time = None
        if log_path:
            log_file = Path(str(log_path))
            if not log_file.is_absolute():
                log_file = repo / log_file
            log_gs, log_train_time = parse_train_log(log_file)

        gs_num = log_gs if log_gs is not None else latest_point_count(run_dir)
        train_time = log_train_time if log_train_time is not None else optional_float(item.get("train_time_s"))
        rows.append(
            {
                "scene": str(item["scene"]),
                "method": str(item["method"]),
                "role": str(item.get("role", "")),
                "compare_to": str(item.get("compare_to", "")),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "gs_num": gs_num,
                "train_time_s": train_time,
                "run_dir": str(run_dir.relative_to(repo) if run_dir.is_relative_to(repo) else run_dir),
            }
        )
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["scene"]), str(row["method"])


def build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row_key(row): row for row in rows}
    comparisons = []
    for row in rows:
        reference_method = str(row.get("compare_to") or "")
        if not reference_method:
            continue
        reference = by_key.get((str(row["scene"]), reference_method))
        if reference is None:
            continue
        deltas = candidate_delta_row(row, reference)
        comparisons.append(
            {
                "scene": row["scene"],
                "method": row["method"],
                "reference": reference_method,
                "delta_psnr": deltas["delta_psnr"],
                "delta_ssim": deltas["delta_ssim"],
                "delta_lpips": deltas["delta_lpips"],
                "delta_gs_num": deltas["delta_gs"],
                "delta_train_time_s": deltas["delta_time"],
                "quality_gain": quality_gain_from_deltas(deltas),
                "gs_growth_band": gs_growth_band(deltas["delta_gs"]),
                "gs_penalty": gs_penalty(deltas["delta_gs"]),
                "qcgi": qcgi_from_deltas(deltas),
            }
        )
    return comparisons


def candidate_delta_row(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "delta_psnr": delta(optional_float(candidate["psnr"]), optional_float(reference["psnr"])),
        "delta_ssim": delta(optional_float(candidate["ssim"]), optional_float(reference["ssim"])),
        "delta_lpips": delta(optional_float(candidate["lpips"]), optional_float(reference["lpips"])),
        "delta_gs": delta(optional_int(candidate["gs_num"]), optional_int(reference["gs_num"])),
        "delta_time": delta(optional_float(candidate["train_time_s"]), optional_float(reference["train_time_s"])),
    }


def quality_candidate_passes(deltas: dict[str, float | int | None]) -> bool:
    d_psnr = deltas["delta_psnr"]
    d_ssim = deltas["delta_ssim"]
    d_lpips = deltas["delta_lpips"]
    if d_psnr is None or d_ssim is None or d_lpips is None:
        return False
    return (
        float(d_psnr) > QUALITY_MIN_DELTA_PSNR
        and float(d_ssim) >= QUALITY_MIN_DELTA_SSIM
        and float(d_lpips) <= QUALITY_MAX_DELTA_LPIPS
    )


def candidate_status(candidate: dict[str, Any] | None, reference: dict[str, Any] | None) -> tuple[str, dict[str, float | int | None]]:
    empty = {
        "delta_psnr": None,
        "delta_ssim": None,
        "delta_lpips": None,
        "delta_gs": None,
        "delta_time": None,
    }
    if candidate is None or reference is None:
        return "missing", empty
    deltas = candidate_delta_row(candidate, reference)
    return ("positive" if quality_candidate_passes(deltas) else "boundary"), deltas


def build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene"]), {})[str(row["method"])] = row

    recommendations = []
    for scene in sorted(by_scene):
        methods = by_scene[scene]
        weighted_i050 = methods.get("weighted_i050")
        weighted_i075 = methods.get("weighted_i075")
        dino_i050 = methods.get("dino_i050")
        budget_pick = "weighted_i050" if weighted_i050 else ("dino_i050" if dino_i050 else "")
        quality_pick = budget_pick
        qcgi_pick = budget_pick
        qcgi_pick_score = 0.0
        reason = "没有更激进质量档通过门槛，保留当前 i0.50 规则。"
        qcgi_reason = "没有候选质量档取得正 QCGI，保留当前 i0.50 规则。"

        candidate_reports: dict[str, tuple[str, dict[str, float | int | None]]] = {}
        passing_candidates = []
        for method in QUALITY_CANDIDATE_METHODS:
            candidate = methods.get(method)
            status, deltas = candidate_status(candidate, weighted_i050)
            candidate_reports[method] = (status, deltas)
            if candidate is not None and status == "positive":
                passing_candidates.append((float(candidate["psnr"]), method, candidate))
            score = qcgi_from_deltas(deltas)
            if candidate is not None and score is not None and score > qcgi_pick_score:
                qcgi_pick = method
                qcgi_pick_score = score
                qcgi_reason = "{} 的 QCGI 最高且为正，质量收益足以覆盖容量代价。".format(method)

        if passing_candidates:
            _, quality_pick, _ = max(passing_candidates, key=lambda item: (item[0], item[1]))
            reason = "{} 在候选质量档中 PSNR 最高，且 SSIM/LPIPS 未明显回落。".format(quality_pick)

        i075_status, i075_deltas = candidate_reports.get(
            "weighted_i075",
            candidate_status(weighted_i075, weighted_i050),
        )
        i090_status, i090_deltas = candidate_reports.get(
            "weighted_i090",
            candidate_status(methods.get("weighted_i090"), weighted_i050),
        )

        recommendations.append(
            {
                "scene": scene,
                "budget_pick": budget_pick,
                "quality_pick": quality_pick,
                "qcgi_pick": qcgi_pick,
                "qcgi_pick_score": qcgi_pick_score,
                "i075_status": i075_status,
                "delta_psnr_i075_vs_i050": i075_deltas["delta_psnr"],
                "delta_ssim_i075_vs_i050": i075_deltas["delta_ssim"],
                "delta_lpips_i075_vs_i050": i075_deltas["delta_lpips"],
                "delta_gs_i075_vs_i050": i075_deltas["delta_gs"],
                "delta_train_time_i075_vs_i050": i075_deltas["delta_time"],
                "qcgi_i075_vs_i050": qcgi_from_deltas(i075_deltas),
                "gs_growth_band_i075_vs_i050": gs_growth_band(i075_deltas["delta_gs"]),
                "i090_status": i090_status,
                "delta_psnr_i090_vs_i050": i090_deltas["delta_psnr"],
                "delta_ssim_i090_vs_i050": i090_deltas["delta_ssim"],
                "delta_lpips_i090_vs_i050": i090_deltas["delta_lpips"],
                "delta_gs_i090_vs_i050": i090_deltas["delta_gs"],
                "delta_train_time_i090_vs_i050": i090_deltas["delta_time"],
                "qcgi_i090_vs_i050": qcgi_from_deltas(i090_deltas),
                "gs_growth_band_i090_vs_i050": gs_growth_band(i090_deltas["delta_gs"]),
                "reason": reason,
                "qcgi_reason": qcgi_reason,
            }
        )
    return recommendations


def build_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    averages = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        avg: dict[str, Any] = {"method": method, "scene_count": len(method_rows)}
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            values = [float(row[key]) for row in method_rows if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = sum(values) / len(values) if values else None
        averages.append(avg)
    return averages


def lookup_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def build_recommendation_averages(
    rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = lookup_rows(rows)
    pick_columns = (
        ("budget_pick", "budget_pick"),
        ("quality_pick", "quality_pick"),
        ("qcgi_pick", "qcgi_pick"),
    )
    averages = []
    for column, selector in pick_columns:
        picked_rows = []
        for recommendation in recommendations:
            method = str(recommendation.get(column) or "")
            if not method:
                continue
            row = by_key.get((str(recommendation["scene"]), method))
            if row is not None:
                picked_rows.append(row)
        avg: dict[str, Any] = {"selector": selector, "scene_count": len(picked_rows)}
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            values = [float(row[key]) for row in picked_rows if row[key] not in (None, "")]
            avg["avg_{}".format(key)] = sum(values) / len(values) if values else None
        averages.append(avg)
    return averages


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 0001 DINO weighted candidates.")
    parser.add_argument("--catalog", type=Path, default=Path("configs/experiments/0001_weighted_candidate_catalog.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/0001/weighted_candidate_summary"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    catalog_path = args.catalog if args.catalog.is_absolute() else repo / args.catalog
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = read_yaml(catalog_path)
    summary = collect_rows(catalog, repo)
    comparisons = build_comparisons(summary)
    recommendations = build_recommendations(summary)
    averages = build_averages(summary)
    recommendation_averages = build_recommendation_averages(summary, recommendations)

    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "comparisons.csv", comparisons)
    write_csv(output_dir / "recommendations.csv", recommendations)
    write_csv(output_dir / "averages.csv", averages)
    write_csv(output_dir / "recommendation_averages.csv", recommendation_averages)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "comparisons.json", comparisons)
    write_json(output_dir / "recommendations.json", recommendations)
    write_json(output_dir / "averages.json", averages)
    write_json(output_dir / "recommendation_averages.json", recommendation_averages)

    print("Wrote {}".format(output_dir / "summary.csv"))
    print("Wrote {}".format(output_dir / "recommendations.csv"))
    print("Wrote {}".format(output_dir / "recommendation_averages.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
