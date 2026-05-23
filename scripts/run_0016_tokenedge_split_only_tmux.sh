#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0016_te_split_g0"
SESSION_G1="0016_te_split_g1"
SESSION_MERGE="0016_te_split_merge"
BASE_OUT="output/0016/tokenedge_split_only_after_baseline"
SWITCH_ITERS="${SWITCH_ITERS:-16000 18000 20000}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0016_tokenedge_split_only_tmux.sh start
  scripts/run_0016_tokenedge_split_only_tmux.sh worker-g0
  scripts/run_0016_tokenedge_split_only_tmux.sh worker-g1
  scripts/run_0016_tokenedge_split_only_tmux.sh merge

Environment:
  SWITCH_ITERS  Space-separated baseline start iterations. Default: "16000 18000 20000".
EOF
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local scenes=("$@")

  echo "[0016] GPU $gpu scenes: ${scenes[*]} switch_iters: $SWITCH_ITERS"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0016_tokenedge_split_only_eval.py \
    --output-root "$out_root" \
    --scenes "${scenes[@]}" \
    --switch-iterations $SWITCH_ITERS
}

merge_results() {
  python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("output/0016/tokenedge_split_only_after_baseline")
groups = [base / "mip_g0", base / "mip_g1"]
combined = base / "mipnerf360_combined"
combined.mkdir(parents=True, exist_ok=True)

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

summary = []
comparison = []
for group in groups:
    summary.extend(read_csv(group / "summary.csv"))
    comparison.extend(read_csv(group / "comparison_start_to_final.csv"))

scene_order = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
scene_rank = {scene: idx for idx, scene in enumerate(scene_order)}
summary.sort(key=lambda row: (scene_rank.get(row["scene"], 999), int(row["switch_iteration"]), row["stage"]))
comparison.sort(key=lambda row: (scene_rank.get(row["scene"], 999), int(row["switch_iteration"])))

write_csv(combined / "summary.csv", summary)
write_csv(combined / "comparison_start_to_final.csv", comparison)
(combined / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(combined / "comparison_start_to_final.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

aggregate = []
for switch in sorted({int(row["switch_iteration"]) for row in comparison}):
    rows = [row for row in comparison if int(row["switch_iteration"]) == switch]
    agg = {"switch_iteration": switch, "scene_count": len(rows)}
    for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        agg["avg_" + key] = sum(values) / len(values) if values else None
    aggregate.append(agg)
write_csv(combined / "aggregate_by_switch.csv", aggregate)
(combined / "aggregate_by_switch.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({"rows": len(summary), "comparisons": len(comparison), "aggregate": aggregate}, indent=2, ensure_ascii=False))
PY
}

start_sessions() {
  mkdir -p output/0016/debug_logs
  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export SWITCH_ITERS='$SWITCH_ITERS' && bash scripts/run_0016_tokenedge_split_only_tmux.sh worker-g0 2>&1 | tee output/0016/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export SWITCH_ITERS='$SWITCH_ITERS' && bash scripts/run_0016_tokenedge_split_only_tmux.sh worker-g1 2>&1 | tee output/0016/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0016_tokenedge_split_only_tmux.sh merge 2>&1 | tee output/0016/debug_logs/${SESSION_MERGE}.log"
  fi
  tmux ls | grep '0016_te_split' || true
}

case "${1:-}" in
  start)
    start_sessions
    ;;
  worker-g0)
    run_worker 0 "$BASE_OUT/mip_g0" bicycle flowers garden stump treehill
    ;;
  worker-g1)
    run_worker 1 "$BASE_OUT/mip_g1" room counter kitchen bonsai
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
