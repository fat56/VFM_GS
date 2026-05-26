#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_0001_fastgs_big_eval import (
    collect_row,
    render_and_metrics,
    scene_overrides,
    train_baseline,
)


DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
DEFAULT_CONFIG = "configs/experiments/0022_taming_importance_inverse_prune_fastgs_big_full9.yaml"
DEFAULT_BASELINE = "output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv"
DEFAULT_METHOD = "taming_importance_inverse_prune_densify100"
DEFAULT_RUN_NAME = "taming_importance_inverse_prune_30k_r_auto"


def read_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["scene"]: row for row in csv.DictReader(handle) if row.get("dataset") == "mipnerf360"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(rows: list[dict[str, object]], baseline: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    comparisons = []
    for row in rows:
        scene = str(row["scene"])
        base = baseline[scene]
        comparisons.append(
            {
                "scene": scene,
                "method": row["method"],
                "baseline_method": base["method"],
                "delta_psnr": float(row["psnr"]) - float(base["psnr"]),
                "delta_ssim": float(row["ssim"]) - float(base["ssim"]),
                "delta_lpips": float(row["lpips"]) - float(base["lpips"]),
                "delta_gs_num": int(row["gs_num"]) - int(base["gs_num"]),
                "delta_train_time_s": float(row["train_time_s"]) - float(base["train_time_s"]),
                "baseline_psnr": base["psnr"],
                "target_psnr": row["psnr"],
                "baseline_ssim": base["ssim"],
                "target_ssim": row["ssim"],
                "baseline_lpips": base["lpips"],
                "target_lpips": row["lpips"],
                "baseline_gs_num": base["gs_num"],
                "target_gs_num": row["gs_num"],
                "baseline_train_time_s": base["train_time_s"],
                "target_train_time_s": row["train_time_s"],
                "run_dir": row["run_dir"],
            }
        )
    return comparisons


def aggregate(rows: list[dict[str, object]], prefix: str = "avg_") -> dict[str, object]:
    out: dict[str, object] = {"scene_count": len(rows)}
    if not rows:
        return out
    for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        out[prefix + key] = sum(values) / len(values) if values else None
    return out


def aggregate_comparison(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"scene_count": len(rows)}
    for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num", "delta_train_time_s"):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        out["avg_" + key] = sum(values) / len(values) if values else None
    return out


def write_outputs(output_root: Path, rows: list[dict[str, object]], baseline: dict[str, dict[str, str]]) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    comparisons = build_comparison(rows, baseline)
    write_csv(output_root / "comparison_vs_fastgs_big_baseline.csv", comparisons)
    (output_root / "comparison_vs_fastgs_big_baseline.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    averages = [{**{"method": rows[0]["method"] if rows else ""}, **aggregate(rows)}]
    write_csv(output_root / "averages.csv", averages)
    (output_root / "averages.json").write_text(json.dumps(averages, indent=2, ensure_ascii=False), encoding="utf-8")

    comp_aggregate = [aggregate_comparison(comparisons)]
    write_csv(output_root / "aggregate_vs_fastgs_big_baseline.csv", comp_aggregate)
    (output_root / "aggregate_vs_fastgs_big_baseline.json").write_text(
        json.dumps(comp_aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0022 Taming-importance inverse-prune FastGS big ablation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/mipnerf360"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--baseline-summary", type=Path, default=Path(DEFAULT_BASELINE))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--no-scene-overrides", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = read_baseline(args.baseline_summary)
    repo = Path.cwd()
    rows: list[dict[str, object]] = []
    use_overrides = not args.no_scene_overrides

    for scene in args.scenes:
        scene_dir = args.dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)
        if scene not in baseline:
            raise KeyError("Missing baseline row for scene {!r}".format(scene))

        overrides = scene_overrides("mipnerf360", scene, use_overrides)
        run_dir = args.output_root / scene / args.run_name
        log_dir = args.output_root / scene / "logs" / args.run_name

        print("[0022] {} taming-importance inverse-prune FastGS big train/render/metrics".format(scene), flush=True)
        train_baseline(scene_dir, scene, args, repo, run_dir, log_dir, overrides)
        render_and_metrics(run_dir, log_dir, repo, overrides)
        rows.append(collect_row("mipnerf360", scene, args.method_name, run_dir, log_dir))
        write_outputs(args.output_root, rows, baseline)

    write_outputs(args.output_root, rows, baseline)
    print("Wrote {}".format(args.output_root / "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
