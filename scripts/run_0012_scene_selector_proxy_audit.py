#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
QUALITY_WEIGHTS = {"psnr": 1.0, "ssim": 20.0, "lpips": 5.0}
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04

BASELINE = Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv")
CURVE = Path("output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv")
FULL_I050 = Path("output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary.csv")
UNTIL8000 = Path("output/0010/descriptor_i050_until8000_full9_combined/comparison_vs_phase0.csv")
UNTIL4000 = Path("output/0011/descriptor_i050_until4000/mipnerf360_combined/comparison_vs_phase0.csv")
OUT_DIR = Path("output/0012/scene_selector_proxy_audit")


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


@dataclass(frozen=True)
class Rule:
    feature: str
    direction: str
    threshold: float

    def matches(self, features: dict[str, float]) -> bool:
        value = features[self.feature]
        if self.direction == ">=":
            return value >= self.threshold
        return value <= self.threshold

    @property
    def label(self) -> str:
        return f"{self.feature} {self.direction} {self.threshold:.6g}"


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique
    thresholds = [unique[0] - 1e-9, unique[-1] + 1e-9]
    thresholds.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    return thresholds


def evaluate_selection(name: str, selected: set[str], u8000: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = []
    for scene in SCENES:
        enabled = scene in selected
        comp = u8000[scene]
        rows.append(
            {
                "scene": scene,
                "enabled": enabled,
                "delta_psnr": as_float(comp, "delta_psnr") if enabled else 0.0,
                "delta_ssim": as_float(comp, "delta_ssim") if enabled else 0.0,
                "delta_lpips": as_float(comp, "delta_lpips") if enabled else 0.0,
                "delta_gs_num": as_float(comp, "delta_gs_num") if enabled else 0.0,
                "qcgi": as_float(comp, "qcgi") if enabled else 0.0,
            }
        )
    return {
        "policy": name,
        "enabled_count": len(selected),
        "enabled_scenes": " ".join(scene for scene in SCENES if scene in selected),
        "avg_delta_psnr": avg([row["delta_psnr"] for row in rows]),
        "avg_delta_ssim": avg([row["delta_ssim"] for row in rows]),
        "avg_delta_lpips": avg([row["delta_lpips"] for row in rows]),
        "avg_delta_gs_num": avg([row["delta_gs_num"] for row in rows]),
        "avg_qcgi": avg([row["qcgi"] for row in rows]),
    }


def best_rule(
    train_scenes: list[str],
    features: dict[str, dict[str, float]],
    u8000: dict[str, dict[str, object]],
    feature_names: list[str],
) -> Rule:
    best: tuple[float, int, str, Rule] | None = None
    for feature in feature_names:
        thresholds = candidate_thresholds([features[scene][feature] for scene in train_scenes])
        for threshold in thresholds:
            for direction in (">=", "<="):
                rule = Rule(feature, direction, threshold)
                selected = {scene for scene in train_scenes if rule.matches(features[scene])}
                score = evaluate_selection("candidate", selected, u8000)["avg_qcgi"]
                tie_break = -abs(len(selected) - (len(train_scenes) / 2.0))
                key = (float(score), int(tie_break * 1000), rule.label)
                if best is None or key > best[:3]:
                    best = (float(score), int(tie_break * 1000), rule.label, rule)
    assert best is not None
    return best[3]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = {row["scene"]: row for row in read_rows(BASELINE)}
    full_rows = {row["scene"]: row for row in read_rows(FULL_I050)}
    u8000 = {row["scene"]: row for row in read_rows(UNTIL8000)}
    u4000 = {row["scene"]: row for row in read_rows(UNTIL4000)}

    curve: dict[str, dict[int, dict[str, str]]] = {scene: {} for scene in SCENES}
    for row in read_rows(CURVE):
        scene = row["scene"]
        if scene in curve:
            curve[scene][int(row["iteration"])] = row

    full_qcgi = {scene: qcgi(full_rows[scene], baseline[scene]) for scene in SCENES}
    target_positive = {scene for scene in SCENES if as_float(u8000[scene], "qcgi") > 0.0}

    feature_rows: list[dict[str, object]] = []
    features: dict[str, dict[str, float]] = {}
    for scene in SCENES:
        c2 = curve[scene][2000]
        c4 = curve[scene][4000]
        c8 = curve[scene][8000]
        c20 = curve[scene][20000]
        c30 = curve[scene][30000]
        values = {
            "phase0_4k_psnr": as_float(c4, "psnr"),
            "phase0_4k_ssim": as_float(c4, "ssim"),
            "phase0_4k_lpips": as_float(c4, "lpips"),
            "phase0_4k_gs": as_float(c4, "gs_num"),
            "phase0_8k_psnr": as_float(c8, "psnr"),
            "phase0_8k_ssim": as_float(c8, "ssim"),
            "phase0_8k_lpips": as_float(c8, "lpips"),
            "phase0_8k_gs": as_float(c8, "gs_num"),
            "phase0_4k_to_8k_psnr_gain": as_float(c8, "psnr") - as_float(c4, "psnr"),
            "phase0_4k_to_8k_gs_gain": as_float(c8, "gs_num") - as_float(c4, "gs_num"),
            "phase0_20k_to_30k_psnr_gain": as_float(c30, "psnr") - as_float(c20, "psnr"),
            "phase0_20k_to_30k_gs_gain": as_float(c30, "gs_num") - as_float(c20, "gs_num"),
            "phase0_30k_psnr": as_float(c30, "psnr"),
            "phase0_30k_ssim": as_float(c30, "ssim"),
            "phase0_30k_lpips": as_float(c30, "lpips"),
            "phase0_30k_gs": as_float(c30, "gs_num"),
            "full_i050_qcgi": full_qcgi[scene],
            "until4000_qcgi": as_float(u4000[scene], "qcgi"),
        }
        features[scene] = values
        feature_rows.append(
            {
                "scene": scene,
                "target_until8000_positive": scene in target_positive,
                "until8000_qcgi": as_float(u8000[scene], "qcgi"),
                **values,
            }
        )

    feature_names = [
        "phase0_4k_psnr",
        "phase0_4k_ssim",
        "phase0_4k_lpips",
        "phase0_4k_gs",
        "phase0_8k_psnr",
        "phase0_8k_ssim",
        "phase0_8k_lpips",
        "phase0_8k_gs",
        "phase0_4k_to_8k_psnr_gain",
        "phase0_4k_to_8k_gs_gain",
        "phase0_20k_to_30k_psnr_gain",
        "phase0_20k_to_30k_gs_gain",
        "phase0_30k_psnr",
        "phase0_30k_ssim",
        "phase0_30k_lpips",
        "phase0_30k_gs",
    ]

    policies = [
        evaluate_selection("all_until8000", set(SCENES), u8000),
        evaluate_selection("oracle_until8000_qcgi_positive", target_positive, u8000),
        evaluate_selection("full_i050_qcgi_positive", {scene for scene in SCENES if full_qcgi[scene] > 0.0}, u8000),
        evaluate_selection("until4000_qcgi_positive", {scene for scene in SCENES if as_float(u4000[scene], "qcgi") > 0.0}, u8000),
    ]

    in_sample_rule = best_rule(SCENES, features, u8000, feature_names)
    policies.append(
        evaluate_selection(
            "best_phase0_curve_single_threshold_in_sample:" + in_sample_rule.label,
            {scene for scene in SCENES if in_sample_rule.matches(features[scene])},
            u8000,
        )
    )

    loocv_predictions = []
    loocv_selected: set[str] = set()
    for heldout in SCENES:
        train = [scene for scene in SCENES if scene != heldout]
        rule = best_rule(train, features, u8000, feature_names)
        enabled = rule.matches(features[heldout])
        if enabled:
            loocv_selected.add(heldout)
        loocv_predictions.append(
            {
                "scene": heldout,
                "target_until8000_positive": heldout in target_positive,
                "enabled": enabled,
                "rule": rule.label,
                "heldout_qcgi": as_float(u8000[heldout], "qcgi") if enabled else 0.0,
            }
        )
    policies.append(evaluate_selection("loocv_phase0_curve_single_threshold", loocv_selected, u8000))

    write_csv(OUT_DIR / "feature_table.csv", feature_rows, list(feature_rows[0].keys()))
    write_csv(OUT_DIR / "policy_summary.csv", policies, list(policies[0].keys()))
    write_csv(OUT_DIR / "loocv_predictions.csv", loocv_predictions, list(loocv_predictions[0].keys()))

    summary = {
        "target_positive_scenes": [scene for scene in SCENES if scene in target_positive],
        "target_negative_scenes": [scene for scene in SCENES if scene not in target_positive],
        "best_in_sample_rule": in_sample_rule.label,
        "policies": policies,
    }
    (OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
