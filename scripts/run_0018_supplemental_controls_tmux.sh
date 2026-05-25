#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0018_controls_g0"
SESSION_G1="0018_controls_g1"
SESSION_MERGE="0018_controls_merge"
BASE_OUT="output/0018/supplemental_controls"
SCENES_G0="${SCENES_G0:-room counter}"
SCENES_G1="${SCENES_G1:-kitchen bonsai}"
VARIANTS="${VARIANTS:-rgb_fastgs_extra_fulltrain desc16k21k_prune35k}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0018_supplemental_controls_tmux.sh start
  scripts/run_0018_supplemental_controls_tmux.sh worker-g0
  scripts/run_0018_supplemental_controls_tmux.sh worker-g1
  scripts/run_0018_supplemental_controls_tmux.sh merge

Environment:
  SCENES_G0  Space-separated GPU0 scenes. Default: "room counter".
  SCENES_G1  Space-separated GPU1 scenes. Default: "kitchen bonsai".
  VARIANTS   Space-separated controls. Default: "rgb_fastgs_extra_fulltrain desc16k21k_prune35k".
EOF
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local scenes=("$@")

  echo "[0018 controls] GPU $gpu scenes: ${scenes[*]}"
  echo "[0018 controls] variants: $VARIANTS"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0018_supplemental_controls_eval.py \
    --output-root "$out_root" \
    --variants $VARIANTS \
    --scenes "${scenes[@]}"
}

merge_results() {
  python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("output/0018/supplemental_controls")
groups = [base / "mip_g0", base / "mip_g1"]
combined = base / "mipnerf360_indoor_combined"
combined.mkdir(parents=True, exist_ok=True)

scene_order = ["room", "counter", "kitchen", "bonsai"]
scene_rank = {name: idx for idx, name in enumerate(scene_order)}
variant_order = ["rgb_fastgs_extra_fulltrain", "desc16k21k_prune35k"]
variant_rank = {name: idx for idx, name in enumerate(variant_order)}

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
baseline_comparison = []
for group in groups:
    summary.extend(read_csv(group / "summary.csv"))
    comparison.extend(read_csv(group / "comparison_start_to_eval.csv"))
    baseline_comparison.extend(read_csv(group / "comparison_vs_baseline30.csv"))

summary.sort(key=lambda row: (
    variant_rank.get(row["variant"], 999),
    scene_rank.get(row["scene"], 999),
    int(row["iteration"]),
))
comparison.sort(key=lambda row: (
    variant_rank.get(row["variant"], 999),
    int(row["relative_iteration"]),
    scene_rank.get(row["scene"], 999),
))
baseline_comparison.sort(key=lambda row: (
    variant_rank.get(row["variant"], 999),
    int(row["iteration"]),
    scene_rank.get(row["scene"], 999),
))

write_csv(combined / "summary.csv", summary)
write_csv(combined / "comparison_start_to_eval.csv", comparison)
write_csv(combined / "comparison_vs_baseline30.csv", baseline_comparison)
(combined / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(combined / "comparison_start_to_eval.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(combined / "comparison_vs_baseline30.json").write_text(
    json.dumps(baseline_comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

aggregate = []
keys = sorted({(row["variant"], int(row["relative_iteration"])) for row in comparison})
for variant, rel in keys:
    rows = [row for row in comparison if row["variant"] == variant and int(row["relative_iteration"]) == rel]
    if not rows:
        continue
    agg = {
        "variant": variant,
        "method": rows[0]["method"],
        "mode": rows[0]["mode"],
        "relative_iteration": rel,
        "scene_count": len(rows),
    }
    for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
        vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        agg["avg_" + key] = sum(vals) / len(vals) if vals else None
    aggregate.append(agg)

write_csv(combined / "aggregate_start_to_eval.csv", aggregate)
(combined / "aggregate_start_to_eval.json").write_text(
    json.dumps(aggregate, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

baseline_aggregate = []
keys = sorted({(row["variant"], int(row["iteration"])) for row in baseline_comparison})
for variant, iteration in keys:
    rows = [row for row in baseline_comparison if row["variant"] == variant and int(row["iteration"]) == iteration]
    if not rows:
        continue
    agg = {
        "variant": variant,
        "method": rows[0]["method"],
        "mode": rows[0]["mode"],
        "iteration": iteration,
        "baseline_iteration": 30000,
        "scene_count": len(rows),
    }
    for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
        vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        agg["avg_" + key] = sum(vals) / len(vals) if vals else None
    baseline_aggregate.append(agg)

write_csv(combined / "aggregate_vs_baseline30.csv", baseline_aggregate)
(combined / "aggregate_vs_baseline30.json").write_text(
    json.dumps(baseline_aggregate, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(
    {
        "summary_rows": len(summary),
        "comparison_rows": len(comparison),
        "baseline_comparison_rows": len(baseline_comparison),
        "aggregate_rows": len(aggregate),
        "baseline_aggregate_rows": len(baseline_aggregate),
        "expected_summary_rows": 36,
        "expected_comparison_rows": 24,
        "expected_baseline_comparison_rows": 16,
    },
    indent=2,
    ensure_ascii=False,
))
PY
}

start_sessions() {
  mkdir -p output/0018/debug_logs
  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export SCENES_G0='$SCENES_G0' VARIANTS='$VARIANTS' && bash scripts/run_0018_supplemental_controls_tmux.sh worker-g0 2>&1 | tee output/0018/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export SCENES_G1='$SCENES_G1' VARIANTS='$VARIANTS' && bash scripts/run_0018_supplemental_controls_tmux.sh worker-g1 2>&1 | tee output/0018/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0018_supplemental_controls_tmux.sh merge 2>&1 | tee output/0018/debug_logs/${SESSION_MERGE}.log"
  fi
  tmux ls | grep '0018_controls' || true
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
