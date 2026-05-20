#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIGNALS = ("prior", "residual_depth", "residual_inv")
RGB_IOU_KEYS = {
    "prior": "prior_rgb_iou",
    "residual_depth": "residual_depth_rgb_iou",
    "residual_inv": "residual_inv_rgb_iou",
}
EDGE_IOU_KEYS = {
    "prior": "prior_gt_edge_iou",
    "residual_depth": "residual_depth_gt_edge_iou",
    "residual_inv": "residual_inv_gt_edge_iou",
}
RGB_L1_KEYS = {
    "prior": "rgb_l1_prior_topk",
    "residual_depth": "rgb_l1_residual_depth_topk",
    "residual_inv": "rgb_l1_residual_inv_topk",
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pick_signal(row: dict[str, Any], keys: dict[str, str]) -> str:
    return max(SIGNALS, key=lambda signal: (as_float(row, keys[signal]), signal))


def build_per_view_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rgb_oracle = pick_signal(row, RGB_IOU_KEYS)
        edge_proxy = pick_signal(row, EDGE_IOU_KEYS)
        l1_oracle = pick_signal(row, RGB_L1_KEYS)
        prior_rgb_iou = as_float(row, RGB_IOU_KEYS["prior"])
        prior_edge_iou = as_float(row, EDGE_IOU_KEYS["prior"])
        prior_l1 = as_float(row, RGB_L1_KEYS["prior"])
        out.append(
            {
                "dataset": row["dataset"],
                "scene": row["scene"],
                "method": row["method"],
                "view_index": row["view_index"],
                "image_name": row["image_name"],
                "proxy_valid_coverage": as_float(row, "proxy_valid_coverage"),
                "rgb_oracle_pick": rgb_oracle,
                "rgb_oracle_rgb_iou": as_float(row, RGB_IOU_KEYS[rgb_oracle]),
                "rgb_oracle_edge_iou": as_float(row, EDGE_IOU_KEYS[rgb_oracle]),
                "rgb_oracle_rgb_l1_topk": as_float(row, RGB_L1_KEYS[rgb_oracle]),
                "edge_proxy_pick": edge_proxy,
                "edge_proxy_rgb_iou": as_float(row, RGB_IOU_KEYS[edge_proxy]),
                "edge_proxy_edge_iou": as_float(row, EDGE_IOU_KEYS[edge_proxy]),
                "edge_proxy_rgb_l1_topk": as_float(row, RGB_L1_KEYS[edge_proxy]),
                "l1_oracle_pick": l1_oracle,
                "l1_oracle_rgb_iou": as_float(row, RGB_IOU_KEYS[l1_oracle]),
                "l1_oracle_edge_iou": as_float(row, EDGE_IOU_KEYS[l1_oracle]),
                "l1_oracle_rgb_l1_topk": as_float(row, RGB_L1_KEYS[l1_oracle]),
                "prior_rgb_iou": prior_rgb_iou,
                "prior_edge_iou": prior_edge_iou,
                "prior_rgb_l1_topk": prior_l1,
                "rgb_oracle_delta_rgb_iou_vs_prior": as_float(row, RGB_IOU_KEYS[rgb_oracle]) - prior_rgb_iou,
                "edge_proxy_delta_rgb_iou_vs_prior": as_float(row, RGB_IOU_KEYS[edge_proxy]) - prior_rgb_iou,
                "edge_proxy_delta_edge_iou_vs_prior": as_float(row, EDGE_IOU_KEYS[edge_proxy]) - prior_edge_iou,
                "edge_proxy_delta_rgb_l1_vs_prior": as_float(row, RGB_L1_KEYS[edge_proxy]) - prior_l1,
                "edge_proxy_matches_rgb_oracle": edge_proxy == rgb_oracle,
                "l1_oracle_delta_rgb_l1_vs_prior": as_float(row, RGB_L1_KEYS[l1_oracle]) - prior_l1,
                "source_run_dir": row.get("run_dir", ""),
                "source_cache_dir": row.get("cache_dir", ""),
            }
        )
    return out


def summarize_group(rows: list[dict[str, Any]], label: dict[str, str]) -> dict[str, Any]:
    rgb_pick_counts = Counter(str(row["rgb_oracle_pick"]) for row in rows)
    edge_pick_counts = Counter(str(row["edge_proxy_pick"]) for row in rows)
    l1_pick_counts = Counter(str(row["l1_oracle_pick"]) for row in rows)
    total = len(rows)
    summary: dict[str, Any] = {
        **label,
        "view_count": total,
        "avg_proxy_valid_coverage": mean([float(row["proxy_valid_coverage"]) for row in rows]),
        "avg_prior_rgb_iou": mean([float(row["prior_rgb_iou"]) for row in rows]),
        "avg_rgb_oracle_rgb_iou": mean([float(row["rgb_oracle_rgb_iou"]) for row in rows]),
        "avg_rgb_oracle_delta_rgb_iou_vs_prior": mean(
            [float(row["rgb_oracle_delta_rgb_iou_vs_prior"]) for row in rows]
        ),
        "avg_edge_proxy_rgb_iou": mean([float(row["edge_proxy_rgb_iou"]) for row in rows]),
        "avg_edge_proxy_delta_rgb_iou_vs_prior": mean(
            [float(row["edge_proxy_delta_rgb_iou_vs_prior"]) for row in rows]
        ),
        "avg_edge_proxy_edge_iou": mean([float(row["edge_proxy_edge_iou"]) for row in rows]),
        "avg_edge_proxy_delta_edge_iou_vs_prior": mean(
            [float(row["edge_proxy_delta_edge_iou_vs_prior"]) for row in rows]
        ),
        "avg_edge_proxy_delta_rgb_l1_vs_prior": mean(
            [float(row["edge_proxy_delta_rgb_l1_vs_prior"]) for row in rows]
        ),
        "edge_proxy_rgb_oracle_match_rate": mean(
            [1.0 if row["edge_proxy_matches_rgb_oracle"] else 0.0 for row in rows]
        ),
        "avg_l1_oracle_delta_rgb_l1_vs_prior": mean(
            [float(row["l1_oracle_delta_rgb_l1_vs_prior"]) for row in rows]
        ),
    }
    for signal in SIGNALS:
        summary[f"rgb_oracle_pick_{signal}"] = rgb_pick_counts.get(signal, 0)
        summary[f"edge_proxy_pick_{signal}"] = edge_pick_counts.get(signal, 0)
        summary[f"l1_oracle_pick_{signal}"] = l1_pick_counts.get(signal, 0)
    return summary


def build_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[(str(row["dataset"]), str(row["scene"]))].append(row)
        by_dataset[str(row["dataset"])].append(row)

    scene_rows = [
        summarize_group(items, {"dataset": dataset, "scene": scene})
        for (dataset, scene), items in sorted(by_scene.items())
    ]
    dataset_rows = [
        summarize_group(items, {"dataset": dataset})
        for dataset, items in sorted(by_dataset.items())
    ]
    overall = summarize_group(rows, {"dataset": "all", "scene": "all"})
    return scene_rows, dataset_rows, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment 0008 residual orientation-aware gating diagnostics.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input per_view.csv files from residual proxy runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/0008/residual_orientation_gating"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_rows: list[dict[str, str]] = []
    for path in args.inputs:
        input_rows.extend(read_csv(path))
    if not input_rows:
        raise ValueError("No rows loaded from inputs")

    per_view_rows = build_per_view_rows(input_rows)
    scene_rows, dataset_rows, overall = build_summaries(per_view_rows)

    write_csv(args.output_dir / "per_view_orientation.csv", per_view_rows)
    write_csv(args.output_dir / "scene_orientation_summary.csv", scene_rows)
    write_csv(args.output_dir / "dataset_orientation_summary.csv", dataset_rows)
    write_json(args.output_dir / "overall_orientation_summary.json", overall)

    print("Wrote {}".format(args.output_dir / "per_view_orientation.csv"))
    print("Wrote {}".format(args.output_dir / "scene_orientation_summary.csv"))
    print("Wrote {}".format(args.output_dir / "overall_orientation_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
