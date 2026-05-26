#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0020_desc_rescue_g0"
SESSION_G1="0020_desc_rescue_g1"
SESSION_MERGE="0020_desc_rescue_merge"
BASE_OUT="output/0020/descriptor_rescue_clone_guarded_pilot"
SCENES_G0="${SCENES_G0:-garden room}"
SCENES_G1="${SCENES_G1:-stump treehill}"

CONFIG="configs/experiments/0020_descriptor_rescue_clone_guarded_pilot.yaml"
METHOD="descriptor_rescue_rgbgate05_cap20_3k12k_baseline30"
RUN_NAME="descriptor_rescue_rgbgate05_cap20_3k12k_baseline30_r_auto"
VARIANT_KEY="0020_descriptor_rescue_clone_guarded_pilot"
BRANCH_NAME="clone_descriptor_rescue_rgb_gate05_cap20_3k12k_split_prune_fastgs_30k"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh start
  scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh worker-g0
  scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh worker-g1
  scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh merge

Environment:
  SCENES_G0  Space-separated GPU0 scenes. Default: "garden room".
  SCENES_G1  Space-separated GPU1 scenes. Default: "stump treehill".
EOF
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local scenes=("$@")

  echo "[0020] GPU $gpu scenes: ${scenes[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py \
    --output-root "$out_root" \
    --scenes "${scenes[@]}" \
    --config "$CONFIG" \
    --variant-key "$VARIANT_KEY" \
    --method-name "$METHOD" \
    --run-name "$RUN_NAME" \
    --branch-name "$BRANCH_NAME" \
    --log-prefix "0020"
}

merge_results() {
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

base = Path("output/0020/descriptor_rescue_clone_guarded_pilot")
groups = [base / "mip_g0", base / "mip_g1"]
combined = base / "mipnerf360_combined"
combined.mkdir(parents=True, exist_ok=True)

expected_scenes = []
for key, default in (("SCENES_G0", "garden room"), ("SCENES_G1", "stump treehill")):
    expected_scenes.extend(os.environ.get(key, default).split())
scene_rank = {name: idx for idx, name in enumerate(expected_scenes)}

def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def avg_rows(rows, group_key):
    aggregate = []
    for group_value in sorted({int(row[group_key]) for row in rows}):
        subset = [row for row in rows if int(row[group_key]) == group_value]
        if not subset:
            continue
        row = {
            "variant": subset[0]["variant"],
            "method": subset[0]["method"],
            group_key: group_value,
            "scene_count": len(subset),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            values = [float(item[key]) for item in subset if item.get(key) not in (None, "")]
            row["avg_" + key] = sum(values) / len(values) if values else None
        aggregate.append(row)
    return aggregate

summary = []
comparison_15k = []
baseline_curve = []
baseline30 = []
for group in groups:
    summary.extend(read_csv(group / "summary.csv"))
    comparison_15k.extend(read_csv(group / "comparison_15k_to_eval.csv"))
    baseline_curve.extend(read_csv(group / "comparison_vs_baseline_curve.csv"))
    baseline30.extend(read_csv(group / "comparison_vs_baseline30.csv"))

summary.sort(key=lambda row: (scene_rank.get(row["scene"], 999), int(row["iteration"])))
comparison_15k.sort(key=lambda row: (int(row["relative_iteration"]), scene_rank.get(row["scene"], 999)))
baseline_curve.sort(key=lambda row: (int(row["iteration"]), scene_rank.get(row["scene"], 999)))
baseline30.sort(key=lambda row: (int(row["iteration"]), scene_rank.get(row["scene"], 999)))

write_csv(combined / "summary.csv", summary)
write_csv(combined / "comparison_15k_to_eval.csv", comparison_15k)
write_csv(combined / "comparison_vs_baseline_curve.csv", baseline_curve)
write_csv(combined / "comparison_vs_baseline30.csv", baseline30)

(combined / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(combined / "comparison_15k_to_eval.json").write_text(
    json.dumps(comparison_15k, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(combined / "comparison_vs_baseline_curve.json").write_text(
    json.dumps(baseline_curve, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(combined / "comparison_vs_baseline30.json").write_text(
    json.dumps(baseline30, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

aggregate_15k = avg_rows(comparison_15k, "relative_iteration")
aggregate_curve = avg_rows(baseline_curve, "iteration")
aggregate30 = avg_rows(baseline30, "iteration")

write_csv(combined / "aggregate_15k_to_eval.csv", aggregate_15k)
write_csv(combined / "aggregate_vs_baseline_curve.csv", aggregate_curve)
write_csv(combined / "aggregate_vs_baseline30.csv", aggregate30)
(combined / "aggregate_15k_to_eval.json").write_text(
    json.dumps(aggregate_15k, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(combined / "aggregate_vs_baseline_curve.json").write_text(
    json.dumps(aggregate_curve, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(combined / "aggregate_vs_baseline30.json").write_text(
    json.dumps(aggregate30, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

scene_count = len(expected_scenes)
print(json.dumps(
    {
        "summary_rows": len(summary),
        "comparison_15k_rows": len(comparison_15k),
        "baseline_curve_rows": len(baseline_curve),
        "baseline30_rows": len(baseline30),
        "aggregate_15k_rows": len(aggregate_15k),
        "aggregate_curve_rows": len(aggregate_curve),
        "aggregate30_rows": len(aggregate30),
        "expected_summary_rows": scene_count * 4,
        "expected_comparison_15k_rows": scene_count * 3,
        "expected_baseline_curve_rows": scene_count * 2,
        "expected_baseline30_rows": scene_count,
    },
    indent=2,
    ensure_ascii=False,
))
PY
}

start_sessions() {
  mkdir -p output/0020/debug_logs
  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export SCENES_G0='$SCENES_G0' && bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh worker-g0 2>&1 | tee output/0020/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export SCENES_G1='$SCENES_G1' && bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh worker-g1 2>&1 | tee output/0020/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && export SCENES_G0='$SCENES_G0' SCENES_G1='$SCENES_G1' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh merge 2>&1 | tee output/0020/debug_logs/${SESSION_MERGE}.log"
  fi
  tmux ls | grep '0020_desc_rescue' || true
}

case "${1:-}" in
  start)
    start_sessions
    ;;
  worker-g0)
    run_worker 0 "$BASE_OUT/mip_g0" $SCENES_G0
    ;;
  worker-g1)
    run_worker 1 "$BASE_OUT/mip_g1" $SCENES_G1
    ;;
  merge)
    merge_results
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
