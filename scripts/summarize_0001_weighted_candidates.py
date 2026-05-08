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
        comparisons.append(
            {
                "scene": row["scene"],
                "method": row["method"],
                "reference": reference_method,
                "delta_psnr": delta(optional_float(row["psnr"]), optional_float(reference["psnr"])),
                "delta_ssim": delta(optional_float(row["ssim"]), optional_float(reference["ssim"])),
                "delta_lpips": delta(optional_float(row["lpips"]), optional_float(reference["lpips"])),
                "delta_gs_num": delta(optional_int(row["gs_num"]), optional_int(reference["gs_num"])),
                "delta_train_time_s": delta(optional_float(row["train_time_s"]), optional_float(reference["train_time_s"])),
            }
        )
    return comparisons


def i075_passes(i075: dict[str, Any], i050: dict[str, Any]) -> bool:
    d_psnr = delta(optional_float(i075["psnr"]), optional_float(i050["psnr"]))
    d_ssim = delta(optional_float(i075["ssim"]), optional_float(i050["ssim"]))
    d_lpips = delta(optional_float(i075["lpips"]), optional_float(i050["lpips"]))
    if d_psnr is None or d_ssim is None or d_lpips is None:
        return False
    return d_psnr > 0.0 and d_ssim >= -0.0002 and d_lpips <= 0.0002


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
        i075_status = "missing"
        reason = "没有 weighted_i075 结果，保留当前 i0.50 规则。"

        delta_psnr = delta_ssim = delta_lpips = delta_gs = delta_time = None
        if weighted_i075 and weighted_i050:
            delta_psnr = delta(optional_float(weighted_i075["psnr"]), optional_float(weighted_i050["psnr"]))
            delta_ssim = delta(optional_float(weighted_i075["ssim"]), optional_float(weighted_i050["ssim"]))
            delta_lpips = delta(optional_float(weighted_i075["lpips"]), optional_float(weighted_i050["lpips"]))
            delta_gs = delta(optional_int(weighted_i075["gs_num"]), optional_int(weighted_i050["gs_num"]))
            delta_time = delta(optional_float(weighted_i075["train_time_s"]), optional_float(weighted_i050["train_time_s"]))
            if i075_passes(weighted_i075, weighted_i050):
                quality_pick = "weighted_i075"
                i075_status = "positive"
                reason = "i0.75 相比 i0.50 的 PSNR 提升，SSIM/LPIPS 未明显回落。"
            else:
                quality_pick = "weighted_i050"
                i075_status = "boundary"
                reason = "i0.75 未超过 i0.50 的质量门槛，推荐保留 i0.50。"

        recommendations.append(
            {
                "scene": scene,
                "budget_pick": budget_pick,
                "quality_pick": quality_pick,
                "i075_status": i075_status,
                "delta_psnr_i075_vs_i050": delta_psnr,
                "delta_ssim_i075_vs_i050": delta_ssim,
                "delta_lpips_i075_vs_i050": delta_lpips,
                "delta_gs_i075_vs_i050": delta_gs,
                "delta_train_time_i075_vs_i050": delta_time,
                "reason": reason,
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
