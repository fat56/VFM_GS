#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIELDS = (
    "dataset",
    "scene",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "gs_num",
    "train_time_s",
    "run_dir",
    "source_method",
    "source_summary",
)


@dataclass(frozen=True)
class CandidateSource:
    path: Path
    method: str
    datasets: tuple[str, ...]
    label: str


MIPNERF360_SOURCES = (
    CandidateSource(
        Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv"),
        "baseline",
        ("mipnerf360",),
        "fastgs_big baseline",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_auto_topk_full/mipnerf360_combined/summary.csv"),
        "depth_auto_topk",
        ("mipnerf360",),
        "Depth Anything prune-protect auto-topk full",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_topk010_full/mipnerf360_combined/summary.csv"),
        "depth_topk010",
        ("mipnerf360",),
        "Depth Anything prune-protect topk010 full",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000/mipnerf360_combined/summary.csv"),
        "depth_rgb_rerank_start9000",
        ("mipnerf360",),
        "Depth Anything RGB rerank start9000 full",
    ),
    CandidateSource(
        Path("output/0004/late_scene_adaptive_auxiliary/final_only_start24000_weight015_gpu0/summary.csv"),
        "depth_start24000_weight015",
        ("mipnerf360",),
        "Late prune-protect start24000 weight0.15 partial",
    ),
    CandidateSource(
        Path("output/0004/late_scene_adaptive_auxiliary/final_only_start24000_auto_topk005_gpu1/summary.csv"),
        "depth_start24000_topk005",
        ("mipnerf360",),
        "Late prune-protect start24000 auto-topk0.5 partial",
    ),
)


CROSS_SOURCES = (
    CandidateSource(
        Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/db/summary.csv"),
        "baseline",
        ("db",),
        "fastgs_big baseline DB",
    ),
    CandidateSource(
        Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt/summary.csv"),
        "baseline",
        ("tandt",),
        "fastgs_big baseline Tandt",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_auto_topk_cross/combined/summary.csv"),
        "depth_auto_topk",
        ("db", "tandt"),
        "Depth Anything prune-protect auto-topk cross",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_topk010_cross/combined/summary.csv"),
        "depth_topk010",
        ("db", "tandt"),
        "Depth Anything prune-protect topk010 cross",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_topk005/combined/summary.csv"),
        "depth_topk005",
        ("db", "tandt"),
        "Depth Anything prune-protect topk005 partial cross",
    ),
    CandidateSource(
        Path("output/0002/depth_anything_depth_prior_prune_protect_weight015_topk010/combined/summary.csv"),
        "depth_weight015_topk010",
        ("db", "tandt"),
        "Depth Anything prune-protect weight0.15 topk010 partial cross",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def usable_run_dir(row: dict[str, str]) -> bool:
    run_dir = Path(row.get("run_dir", ""))
    return bool(row.get("run_dir")) and run_dir.exists() and (run_dir / "cfg_args").exists()


def normalize_row(row: dict[str, str], source: CandidateSource) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "scene": row["scene"],
        "method": source.method,
        "psnr": row.get("psnr", ""),
        "ssim": row.get("ssim", ""),
        "lpips": row.get("lpips", ""),
        "gs_num": row.get("gs_num", ""),
        "train_time_s": row.get("train_time_s", ""),
        "run_dir": row.get("run_dir", ""),
        "source_method": row.get("method", ""),
        "source_summary": str(source.path),
    }


def build_rows(sources: tuple[CandidateSource, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    manifest: dict[str, Any] = {"sources": [], "skipped": []}
    for source in sources:
        source_record: dict[str, Any] = {
            "label": source.label,
            "path": str(source.path),
            "method": source.method,
            "datasets": list(source.datasets),
            "rows": 0,
            "accepted": 0,
        }
        if not source.path.exists():
            manifest["skipped"].append(
                {"path": str(source.path), "reason": "missing_summary", "label": source.label}
            )
            manifest["sources"].append(source_record)
            continue
        for row in read_csv(source.path):
            source_record["rows"] += 1
            if row.get("dataset") not in source.datasets:
                continue
            if not usable_run_dir(row):
                manifest["skipped"].append(
                    {
                        "path": str(source.path),
                        "dataset": row.get("dataset", ""),
                        "scene": row.get("scene", ""),
                        "method": row.get("method", ""),
                        "reason": "missing_run_dir_or_cfg_args",
                        "run_dir": row.get("run_dir", ""),
                    }
                )
                continue
            normalized = normalize_row(row, source)
            key = (
                str(normalized["dataset"]),
                str(normalized["scene"]),
                str(normalized["method"]),
            )
            if key in rows_by_key:
                manifest["skipped"].append(
                    {
                        "path": str(source.path),
                        "dataset": key[0],
                        "scene": key[1],
                        "method": key[2],
                        "reason": "duplicate_candidate_key",
                    }
                )
                continue
            rows_by_key[key] = normalized
            source_record["accepted"] += 1
        manifest["sources"].append(source_record)

    rows = sorted(rows_by_key.values(), key=lambda row: (row["dataset"], row["scene"], row["method"]))
    manifest["row_count"] = len(rows)
    manifest["scene_count"] = len({(row["dataset"], row["scene"]) for row in rows})
    manifest["method_count"] = len({row["method"] for row in rows})
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment 0006 selector candidate tables.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/0006/validation_selector"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mip_rows, mip_manifest = build_rows(MIPNERF360_SOURCES)
    cross_rows, cross_manifest = build_rows(CROSS_SOURCES)
    all_rows = sorted(
        [*mip_rows, *cross_rows],
        key=lambda row: (row["dataset"], row["scene"], row["method"]),
    )

    write_csv(args.output_dir / "mipnerf360_depth_candidates.csv", mip_rows)
    write_csv(args.output_dir / "cross_depth_candidates.csv", cross_rows)
    write_csv(args.output_dir / "all_depth_candidates.csv", all_rows)

    manifest = {
        "mipnerf360": mip_manifest,
        "cross": cross_manifest,
        "all_row_count": len(all_rows),
    }
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wrote {}".format(args.output_dir / "mipnerf360_depth_candidates.csv"))
    print("Wrote {}".format(args.output_dir / "cross_depth_candidates.csv"))
    print("Wrote {}".format(args.output_dir / "candidate_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
