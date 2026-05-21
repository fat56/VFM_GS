#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_0014_phase0_fingerprint_cross_method_selector import (
    SCENES,
    METHODS,
    best_stump,
    build_phase0_features,
    normalized_distance,
    oracle_predictions,
    policy_summary,
    qcgi,
    read_comparison,
    read_rows,
    standardize,
    write_csv,
)


RESIDUAL_SUMMARY = Path("output/0008/residual_orientation_gating_mipnerf360/scene_orientation_summary.csv")
OUT_DIR = Path("output/0015/residual_proxy_cross_method_selector")


def build_phase0_residual_features() -> tuple[dict[str, dict[str, float]], list[dict[str, object]], list[str]]:
    phase0_features, _, _ = build_phase0_features()
    residual_rows = {row["scene"]: row for row in read_rows(RESIDUAL_SUMMARY)}
    residual_feature_names = [
        key
        for key in next(iter(residual_rows.values())).keys()
        if key not in {"dataset", "scene"}
    ]

    features: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for scene in SCENES:
        values = dict(phase0_features[scene])
        residual = residual_rows[scene]
        for key in residual_feature_names:
            values[f"residual_proxy_{key}"] = float(residual[key])
        features[scene] = values
        rows.append({"scene": scene, **values})

    return features, rows, list(features[SCENES[0]].keys())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features, feature_rows, feature_names = build_phase0_residual_features()
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
            "in_sample_phase0_plus_residual_proxy_stump:" + in_sample_stump.label,
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
                "selector": "loocv_phase0_plus_residual_proxy_stump",
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
                "selector": "loocv_phase0_plus_residual_proxy_nearest_neighbor",
                "scene": heldout,
                "selected_method": nn_method,
                "oracle_method": oracle[heldout],
                "selected_qcgi": qcgi(comparison, heldout, nn_method),
                "oracle_qcgi": qcgi(comparison, heldout, oracle[heldout]),
                "meta": f"nearest={nearest}; distance={normalized_distance(features, heldout, nearest, feature_names, means, stds):.6g}",
            }
        )

    policies.append(policy_summary("loocv_phase0_plus_residual_proxy_stump", loocv_stump_predictions, comparison))
    policies.append(
        policy_summary("loocv_phase0_plus_residual_proxy_nearest_neighbor", loocv_nn_predictions, comparison)
    )

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
        "feature_count": len(feature_names),
        "in_sample_stump": in_sample_stump.label,
        "policies": policies,
    }
    (OUT_DIR / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
