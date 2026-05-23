#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0016_smoke_g0"
SESSION_G1="0016_smoke_g1"
SESSION_MERGE="0016_smoke_merge"
BASE_OUT="output/0016/supplemental_smoke"
SCENES="${SCENES:-kitchen flowers bonsai}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0016_supplemental_smoke_tmux.sh start
  scripts/run_0016_supplemental_smoke_tmux.sh worker-g0
  scripts/run_0016_supplemental_smoke_tmux.sh worker-g1
  scripts/run_0016_supplemental_smoke_tmux.sh merge

Environment:
  SCENES  Space-separated smoke scenes. Default: "kitchen flowers bonsai".
EOF
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local variants=("$@")

  echo "[0016-smoke] GPU $gpu variants: ${variants[*]} scenes: $SCENES"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0016_supplemental_smoke_eval.py \
    --output-root "$out_root" \
    --scenes $SCENES \
    --variants "${variants[@]}"
}

merge_results() {
  python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("output/0016/supplemental_smoke")
groups = [base / "mip_g0", base / "mip_g1"]
combined = base / "mipnerf360_smoke_combined"
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

variant_order = ["desc_clone_16k", "desc_split_16k", "desc_clone_30k", "tokenedge_split_30k"]
scene_order = ["kitchen", "flowers", "bonsai"]
variant_rank = {name: idx for idx, name in enumerate(variant_order)}
scene_rank = {name: idx for idx, name in enumerate(scene_order)}

summary = []
comparison = []
aggregate = []
for group in groups:
    summary.extend(read_csv(group / "summary.csv"))
    comparison.extend(read_csv(group / "comparison_start_to_eval.csv"))
    aggregate.extend(read_csv(group / "aggregate_by_variant.csv"))

summary.sort(
    key=lambda row: (
        variant_rank.get(row["variant"], 999),
        scene_rank.get(row["scene"], 999),
        int(row["relative_iteration"]),
    )
)
comparison.sort(
    key=lambda row: (
        variant_rank.get(row["variant"], 999),
        scene_rank.get(row["scene"], 999),
        int(row["relative_iteration"]),
    )
)

write_csv(combined / "summary.csv", summary)
write_csv(combined / "comparison_start_to_eval.csv", comparison)
(combined / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(combined / "comparison_start_to_eval.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

merged_aggregate = []
for variant in sorted({row["variant"] for row in comparison}, key=lambda name: variant_rank.get(name, 999)):
    for rel in sorted({int(row["relative_iteration"]) for row in comparison if row["variant"] == variant}):
        rows = [row for row in comparison if row["variant"] == variant and int(row["relative_iteration"]) == rel]
        if not rows:
            continue
        agg = {
            "variant": variant,
            "method": rows[0]["method"],
            "backend": rows[0]["backend"],
            "relative_iteration": rel,
            "scene_count": len(rows),
        }
        for key in ("delta_psnr", "delta_ssim", "delta_lpips", "delta_gs_num"):
            vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
            agg["avg_" + key] = sum(vals) / len(vals) if vals else None
        merged_aggregate.append(agg)
write_csv(combined / "aggregate_by_variant.csv", merged_aggregate)
(combined / "aggregate_by_variant.json").write_text(
    json.dumps(merged_aggregate, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps({"rows": len(summary), "comparisons": len(comparison), "aggregate": merged_aggregate}, indent=2, ensure_ascii=False))
PY
}

start_sessions() {
  mkdir -p output/0016/debug_logs
  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export SCENES='$SCENES' && bash scripts/run_0016_supplemental_smoke_tmux.sh worker-g0 2>&1 | tee output/0016/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export SCENES='$SCENES' && bash scripts/run_0016_supplemental_smoke_tmux.sh worker-g1 2>&1 | tee output/0016/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0016_supplemental_smoke_tmux.sh merge 2>&1 | tee output/0016/debug_logs/${SESSION_MERGE}.log"
  fi
  tmux ls | grep '0016_smoke' || true
}

case "${1:-}" in
  start)
    start_sessions
    ;;
  worker-g0)
    run_worker 0 "$BASE_OUT/mip_g0" desc_clone_16k desc_split_16k
    ;;
  worker-g1)
    run_worker 1 "$BASE_OUT/mip_g1" desc_clone_30k tokenedge_split_30k
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
