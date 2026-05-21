#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
METHODS = [
    "phase0",
    "depth_auto_topk",
    "residual_orientation",
    "dino_i050_full",
    "dino_i050_until8000",
    "dino_i050_until4000",
]

CURVE = Path("output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv")
COMPARISON = Path("output/0013/cross_method_scene_oracle/method_comparison_vs_phase0.csv")
OUT_DIR = Path("output/0014/phase0_fingerprint_cross_method_selector")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]


@dataclass(frozen=True)
class Stump:
    feature: str
    threshold: float
    le_method: str
    gt_method: str

    @property
    def label(self) -> str:
        return f"{self.feature} <= {self.threshold:.6g} ? {self.le_method} : {self.gt_method}"

    def predict(self, values: dict[str, float]) -> str:
        return self.le_method if values[self.feature] <= self.threshold else self.gt_method


def method_row(
    comparison: dict[str, dict[str, dict[str, str]]],
    scene: str,
    method: str,
) -> dict[str, str]:
    return comparison[scene][method]


def qcgi(
    comparison: dict[str, dict[str, dict[str, str]]],
    scene: str,
    method: str,
) -> float:
    return as_float(method_row(comparison, scene, method), "qcgi")


def best_method_for_group(
    scenes: list[str],
    comparison: dict[str, dict[str, dict[str, str]]],
) -> str:
    best_method = "phase0"
    best_score = -math.inf
    for method in METHODS:
        score = avg([qcgi(comparison, scene, method) for scene in scenes])
        if score > best_score + 1e-12:
            best_score = score
            best_method = method
    return best_method


def policy_summary(
    name: str,
    predictions: dict[str, str],
    comparison: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    rows = [method_row(comparison, scene, predictions[scene]) for scene in SCENES]
    return {
        "policy": name,
        "avg_delta_psnr": avg([as_float(row, "delta_psnr") for row in rows]),
        "avg_delta_ssim": avg([as_float(row, "delta_ssim") for row in rows]),
        "avg_delta_lpips": avg([as_float(row, "delta_lpips") for row in rows]),
        "avg_delta_gs_num": avg([as_float(row, "delta_gs_num") for row in rows]),
        "avg_qcgi": avg([as_float(row, "qcgi") for row in rows]),
        "selected_methods": " ".join(predictions[scene] for scene in SCENES),
    }


def build_phase0_features() -> tuple[dict[str, dict[str, float]], list[dict[str, object]], list[str]]:
    curve: dict[str, dict[int, dict[str, str]]] = {scene: {} for scene in SCENES}
    for row in read_rows(CURVE):
        scene = row["scene"]
        if scene in curve:
            curve[scene][int(row["iteration"])] = row

    iterations = [2000, 4000, 8000, 16000, 20000, 30000]
    pairs = [(2000, 4000), (4000, 8000), (8000, 16000), (16000, 20000), (20000, 30000)]
    metric_keys = ["psnr", "ssim", "lpips", "gs_num"]
    features: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []

    for scene in SCENES:
        values: dict[str, float] = {}
        for iteration in iterations:
            row = curve[scene][iteration]
            for key in metric_keys:
                values[f"i{iteration}_{key}"] = as_float(row, key)

        for start, end in pairs:
            start_row = curve[scene][start]
            end_row = curve[scene][end]
            for key in metric_keys:
                values[f"i{start}_to_i{end}_{key}_gain"] = as_float(end_row, key) - as_float(start_row, key)
            start_gs = as_float(start_row, "gs_num")
            values[f"i{start}_to_i{end}_gs_ratio"] = as_float(end_row, "gs_num") / start_gs if start_gs else 0.0

        values["i30000_gs_per_psnr"] = values["i30000_gs_num"] / max(values["i30000_psnr"], 1e-6)
        values["i8000_to_i30000_psnr_gain"] = values["i30000_psnr"] - values["i8000_psnr"]
        values["i8000_to_i30000_gs_gain"] = values["i30000_gs_num"] - values["i8000_gs_num"]
        values["i8000_to_i30000_lpips_gain"] = values["i30000_lpips"] - values["i8000_lpips"]

        features[scene] = values
        rows.append({"scene": scene, **values})

    return features, rows, list(features[SCENES[0]].keys())


def read_comparison() -> dict[str, dict[str, dict[str, str]]]:
    comparison: dict[str, dict[str, dict[str, str]]] = {scene: {} for scene in SCENES}
    for row in read_rows(COMPARISON):
        comparison[row["scene"]][row["method"]] = row
    return comparison


def oracle_predictions(comparison: dict[str, dict[str, dict[str, str]]]) -> dict[str, str]:
    predictions = {}
    for scene in SCENES:
        predictions[scene] = max(METHODS, key=lambda method: qcgi(comparison, scene, method))
    return predictions


def best_stump(
    train_scenes: list[str],
    features: dict[str, dict[str, float]],
    feature_names: list[str],
    comparison: dict[str, dict[str, dict[str, str]]],
) -> Stump:
    best: tuple[float, int, str, Stump] | None = None
    for feature in feature_names:
        thresholds = candidate_thresholds([features[scene][feature] for scene in train_scenes])
        for threshold in thresholds:
            le_scenes = [scene for scene in train_scenes if features[scene][feature] <= threshold]
            gt_scenes = [scene for scene in train_scenes if features[scene][feature] > threshold]
            if not le_scenes or not gt_scenes:
                continue
            le_method = best_method_for_group(le_scenes, comparison)
            gt_method = best_method_for_group(gt_scenes, comparison)
            stump = Stump(feature, threshold, le_method, gt_method)
            score = avg(
                [
                    qcgi(comparison, scene, stump.predict(features[scene]))
                    for scene in train_scenes
                ]
            )
            min_leaf_size = min(len(le_scenes), len(gt_scenes))
            key = (score, min_leaf_size, stump.label)
            if best is None or key > best[:3]:
                best = (score, min_leaf_size, stump.label, stump)
    assert best is not None
    return best[3]


def standardize(
    features: dict[str, dict[str, float]],
    train_scenes: list[str],
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature in feature_names:
        values = [features[scene][feature] for scene in train_scenes]
        mean = avg(values)
        variance = avg([(value - mean) ** 2 for value in values])
        means[feature] = mean
        stds[feature] = math.sqrt(variance)
    return means, stds


def normalized_distance(
    features: dict[str, dict[str, float]],
    a: str,
    b: str,
    feature_names: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> float:
    total = 0.0
    for feature in feature_names:
        std = stds[feature]
        if std <= 1e-12:
            continue
        av = (features[a][feature] - means[feature]) / std
        bv = (features[b][feature] - means[feature]) / std
        total += (av - bv) ** 2
    return math.sqrt(total)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features, feature_rows, feature_names = build_phase0_features()
    comparison = read_comparison()
    oracle = oracle_predictions(comparison)

    policies: list[dict[str, object]] = []
    for method in METHODS:
        policies.append(policy_summary(f"fixed_{method}", {scene: method for scene in SCENES}, comparison))
    policies.append(policy_summary("oracle_best_qcgi_per_scene", oracle, comparison))

    in_sample_stump = best_stump(SCENES, features, feature_names, comparison)
    in_sample_predictions = {scene: in_sample_stump.predict(features[scene]) for scene in SCENES}
    policies.append(
        policy_summary(
            "in_sample_phase0_curve_stump:" + in_sample_stump.label,
            in_sample_predictions,
            comparison,
        )
    )

    loocv_stump_predictions: dict[str, str] = {}
    loocv_nn_predictions: dict[str, str] = {}
    prediction_rows: list[dict[str, object]] = []
    for heldout in SCENES:
        train_scenes = [scene for scene in SCENES if scene != heldout]

        stump = best_stump(train_scenes, features, feature_names, comparison)
        stump_method = stump.predict(features[heldout])
        loocv_stump_predictions[heldout] = stump_method
        prediction_rows.append(
            {
                "selector": "loocv_phase0_curve_stump",
                "scene": heldout,
                "selected_method": stump_method,
                "oracle_method": oracle[heldout],
                "selected_qcgi": qcgi(comparison, heldout, stump_method),
                "oracle_qcgi": qcgi(comparison, heldout, oracle[heldout]),
                "meta": stump.label,
            }
        )

        means, stds = standardize(features, train_scenes, feature_names)
        nearest = min(
            train_scenes,
            key=lambda scene: normalized_distance(features, heldout, scene, feature_names, means, stds),
        )
        nn_method = oracle[nearest]
        loocv_nn_predictions[heldout] = nn_method
        prediction_rows.append(
            {
                "selector": "loocv_nearest_neighbor_oracle_method",
                "scene": heldout,
                "selected_method": nn_method,
                "oracle_method": oracle[heldout],
                "selected_qcgi": qcgi(comparison, heldout, nn_method),
                "oracle_qcgi": qcgi(comparison, heldout, oracle[heldout]),
                "meta": f"nearest={nearest}; distance={normalized_distance(features, heldout, nearest, feature_names, means, stds):.6g}",
            }
        )

    policies.append(policy_summary("loocv_phase0_curve_stump", loocv_stump_predictions, comparison))
    policies.append(policy_summary("loocv_nearest_neighbor_oracle_method", loocv_nn_predictions, comparison))

    write_csv(OUT_DIR / "feature_table.csv", feature_rows, ["scene", *feature_names])
    write_csv(
        OUT_DIR / "loocv_predictions.csv",
        prediction_rows,
        ["selector", "scene", "selected_method", "oracle_method", "selected_qcgi", "oracle_qcgi", "meta"],
    )
    write_csv(
        OUT_DIR / "policy_summary.csv",
        policies,
        ["policy", "avg_delta_psnr", "avg_delta_ssim", "avg_delta_lpips", "avg_delta_gs_num", "avg_qcgi", "selected_methods"],
    )

    summary = {
        "oracle_selection": oracle,
        "in_sample_stump": in_sample_stump.label,
        "policies": policies,
    }
    (OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
