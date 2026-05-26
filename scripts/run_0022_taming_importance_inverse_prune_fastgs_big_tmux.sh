#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0022_taming_invprune_g0"
SESSION_G1="0022_taming_invprune_g1"
SESSION_MERGE="0022_taming_invprune_merge"
BASE_OUT="output/0022/taming_importance_inverse_prune_fastgs_big_full9"
SCENES_G0="${SCENES_G0:-garden bicycle flowers room}"
SCENES_G1="${SCENES_G1:-kitchen bonsai stump treehill counter}"

CONFIG="configs/experiments/0022_taming_importance_inverse_prune_fastgs_big_full9.yaml"
METHOD="taming_importance_inverse_prune_densify100"
RUN_NAME="taming_importance_inverse_prune_30k_r_auto"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh start
  scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh worker-g0
  scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh worker-g1
  scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh merge

Environment:
  SCENES_G0  Space-separated GPU0 scenes. Default: "garden bicycle flowers room".
  SCENES_G1  Space-separated GPU1 scenes. Default: "kitchen bonsai stump treehill counter".
EOF
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local scenes=("$@")

  echo "[0022] GPU $gpu scenes: ${scenes[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0022_taming_importance_inverse_prune_fastgs_big_eval.py \
    --output-root "$out_root" \
    --scenes "${scenes[@]}" \
    --config "$CONFIG" \
    --method-name "$METHOD" \
    --run-name "$RUN_NAME" \
    --densification-interval 100
}

merge_results() {
  python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("output/0022/taming_importance_inverse_prune_fastgs_big_full9")
groups = [base / "mip_g0", base / "mip_g1"]
combined = base / "mipnerf360_combined"
combined.mkdir(parents=True, exist_ok=True)

scene_order = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
scene_rank = {scene: idx for idx, scene in enumerate(scene_order)}

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

def average(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None

summary = []
comparison = []
for group in groups:
    summary.extend(read_csv(group / "summary.csv"))
    comparison.extend(read_csv(group / "comparison_vs_fastgs_big_baseline.csv"))

summary.sort(key=lambda row: scene_rank.get(row["scene"], 999))
comparison.sort(key=lambda row: scene_rank.get(row["scene"], 999))

write_csv(combined / "summary.csv", summary)
write_csv(combined / "comparison_vs_fastgs_big_baseline.csv", comparison)

(combined / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(combined / "comparison_vs_fastgs_big_baseline.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

averages = []
if summary:
    averages.append(
        {
            "method": summary[0]["method"],
            "scene_count": len(summary),
            "avg_psnr": average(summary, "psnr"),
            "avg_ssim": average(summary, "ssim"),
            "avg_lpips": average(summary, "lpips"),
            "avg_gs_num": average(summary, "gs_num"),
            "avg_train_time_s": average(summary, "train_time_s"),
        }
    )
write_csv(combined / "averages.csv", averages)
(combined / "averages.json").write_text(json.dumps(averages, indent=2, ensure_ascii=False), encoding="utf-8")

aggregate = []
if comparison:
    aggregate.append(
        {
            "scene_count": len(comparison),
            "avg_delta_psnr": average(comparison, "delta_psnr"),
            "avg_delta_ssim": average(comparison, "delta_ssim"),
            "avg_delta_lpips": average(comparison, "delta_lpips"),
            "avg_delta_gs_num": average(comparison, "delta_gs_num"),
            "avg_delta_train_time_s": average(comparison, "delta_train_time_s"),
        }
    )
write_csv(combined / "aggregate_vs_fastgs_big_baseline.csv", aggregate)
(combined / "aggregate_vs_fastgs_big_baseline.json").write_text(
    json.dumps(aggregate, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(
    {
        "summary_rows": len(summary),
        "comparison_rows": len(comparison),
        "expected_summary_rows": 9,
        "expected_comparison_rows": 9,
    },
    indent=2,
    ensure_ascii=False,
))
PY
}

start_sessions() {
  mkdir -p output/0022/debug_logs
  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export SCENES_G0='$SCENES_G0' && bash scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh worker-g0 2>&1 | tee output/0022/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export SCENES_G1='$SCENES_G1' && bash scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh worker-g1 2>&1 | tee output/0022/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh merge 2>&1 | tee output/0022/debug_logs/${SESSION_MERGE}.log"
  fi
  tmux ls | grep '0022_taming_invprune' || true
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
