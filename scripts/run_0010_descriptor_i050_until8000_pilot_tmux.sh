#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_G0="0010_i050_u8000_g0"
SESSION_G1="0010_i050_u8000_g1"
SESSION_MERGE="0010_i050_u8000_merge"
RUN_NAME="descriptor_i050_until8000_30k_r_auto"
BASE_OUT="output/0010/descriptor_i050_until8000_pilot"
CONFIG="configs/experiments/0010_descriptor_i050_active_until8000.yaml"
METHOD="descriptor_i050_until8000"
BASELINE="output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv"
FULL_I050="output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary.csv"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh start
  scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh worker-g0
  scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh worker-g1
  scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh merge

Environment:
  WAIT_PIDS  Optional whitespace-separated PIDs to wait for before worker starts.
EOF
}

existing_pids_for() {
  local group="$1"
  ps -eo pid=,cmd= \
    | awk -v group="$group" '
      $0 ~ "scripts/run_0001_fastgs_big_eval.py" &&
      $0 ~ "output/0010/descriptor_i050_until8000_pilot/" group &&
      $0 !~ "awk" { print $1 }
    ' \
    | tr '\n' ' '
}

wait_for_pids() {
  local pids="${WAIT_PIDS:-}"
  if [[ -z "$pids" ]]; then
    return
  fi
  echo "[0010-u8000] waiting for existing foreground PIDs: $pids"
  for pid in $pids; do
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
  done
}

results_path() {
  local out_root="$1"
  local scene="$2"
  printf "%s/%s/%s/results.json" "$out_root" "$scene" "$RUN_NAME"
}

archive_incomplete() {
  local out_root="$1"
  local scene="$2"
  local run_dir="$out_root/$scene/$RUN_NAME"
  local log_dir="$out_root/$scene/logs/$RUN_NAME"
  local result_file
  result_file="$(results_path "$out_root" "$scene")"
  if [[ -d "$run_dir" && ! -s "$result_file" ]]; then
    local stamp archive_root
    stamp="$(date +%Y%m%d_%H%M%S)"
    archive_root="output/0010/debug_artifacts/interrupted_until8000_${stamp}/$(basename "$out_root")/$scene"
    mkdir -p "$archive_root"
    mv "$run_dir" "$archive_root/run_dir"
    if [[ -d "$log_dir" ]]; then
      mkdir -p "$archive_root/logs_parent"
      mv "$log_dir" "$archive_root/logs_parent/$RUN_NAME"
    fi
    echo "[0010-u8000] archived incomplete $scene to $archive_root"
  fi
}

run_worker() {
  local gpu="$1"
  local out_root="$2"
  shift 2
  local scenes=("$@")

  wait_for_pids

  local missing=()
  for scene in "${scenes[@]}"; do
    if [[ ! -s "$(results_path "$out_root" "$scene")" ]]; then
      archive_incomplete "$out_root" "$scene"
      missing+=("$scene")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[0010-u8000] no missing scenes for $out_root"
    return
  fi

  echo "[0010-u8000] GPU $gpu running scenes: ${missing[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --active python scripts/run_0001_fastgs_big_eval.py \
    --dataset-name mipnerf360 \
    --dataset-root datasets/mipnerf360 \
    --output-root "$out_root" \
    --scenes "${missing[@]}" \
    --train-images images \
    --iterations 30000 \
    --resolution -1 \
    --variant fastgs_big \
    --densification-interval 100 \
    --config "$CONFIG" \
    --method-name "$METHOD" \
    --run-name "$RUN_NAME" \
    --vfm-cache-template 'output/0001/vfm_cache/{scene}_dinov2_vits14'
}

merge_results() {
  python - <<'PY'
import csv
import json
import re
from pathlib import Path

base_out = Path("output/0010/descriptor_i050_until8000_pilot")
run_name = "descriptor_i050_until8000_30k_r_auto"
method = "descriptor_i050_until8000"
baseline_path = Path("output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv")
full_i050_path = Path("output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary.csv")
combined = base_out / "mipnerf360_pilot_combined"
combined.mkdir(parents=True, exist_ok=True)

scenes = ["bicycle", "garden", "stump", "counter", "kitchen", "bonsai"]
groups = {
    "bicycle": "mip_g0",
    "garden": "mip_g0",
    "stump": "mip_g0",
    "counter": "mip_g1",
    "kitchen": "mip_g1",
    "bonsai": "mip_g1",
}

QUALITY_WEIGHTS = {"psnr": 1.0, "ssim": 20.0, "lpips": 5.0}
GS_UNIT = 10_000.0
GS_SOFT_BUDGET = 100_000.0
GS_PENALTY_PER_10K = 0.01
GS_HEAVY_PENALTY_PER_10K = 0.04

def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def read_results(run_dir: Path):
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    key = sorted(data.keys())[-1]
    return data[key]

def parse_train_log(log_path: Path):
    gs = None
    train_time = None
    if not log_path.exists():
        return gs, train_time
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gs_matches = re.findall(r"Gaussian number:\s*(\d+)", text)
    time_matches = re.findall(r"Training time:\s*([0-9.]+)", text)
    if gs_matches:
        gs = int(gs_matches[-1])
    if time_matches:
        train_time = float(time_matches[-1])
    return gs, train_time

def as_float(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)

def as_int(row, key):
    value = row.get(key)
    return None if value in (None, "") else int(float(value))

def quality_gain(row, ref):
    d_psnr = as_float(row, "psnr") - as_float(ref, "psnr")
    d_ssim = as_float(row, "ssim") - as_float(ref, "ssim")
    d_lpips = as_float(row, "lpips") - as_float(ref, "lpips")
    return (
        QUALITY_WEIGHTS["psnr"] * d_psnr
        + QUALITY_WEIGHTS["ssim"] * d_ssim
        - QUALITY_WEIGHTS["lpips"] * d_lpips
    )

def gs_penalty(delta_gs):
    growth = max(0.0, float(delta_gs))
    soft_growth = min(growth, GS_SOFT_BUDGET)
    heavy_growth = max(0.0, growth - GS_SOFT_BUDGET)
    return (
        GS_PENALTY_PER_10K * (soft_growth / GS_UNIT)
        + GS_HEAVY_PENALTY_PER_10K * (heavy_growth / GS_UNIT)
    )

def qcgi(row, ref):
    delta_gs = as_int(row, "gs_num") - as_int(ref, "gs_num")
    return quality_gain(row, ref) - gs_penalty(delta_gs)

def avg(values):
    values = [float(v) for v in values if v not in (None, "")]
    return sum(values) / len(values) if values else None

rows = []
missing = []
for scene in scenes:
    group = groups[scene]
    run_dir = base_out / group / scene / run_name
    result_path = run_dir / "results.json"
    if not result_path.exists():
        missing.append(scene)
        continue
    metrics = read_results(run_dir)
    gs, train_time = parse_train_log(base_out / group / scene / "logs" / run_name / "train.log")
    rows.append({
        "dataset": "mipnerf360",
        "scene": scene,
        "method": method,
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gs_num": gs,
        "train_time_s": train_time,
        "run_dir": str(run_dir),
    })

fields = ["dataset", "scene", "method", "psnr", "ssim", "lpips", "gs_num", "train_time_s", "run_dir"]
write_csv(combined / "summary.csv", rows, fields)
(combined / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

baseline = {row["scene"]: row for row in read_rows(baseline_path)}
full_i050 = {row["scene"]: row for row in read_rows(full_i050_path)}

comparison_fields = [
    "dataset",
    "scene",
    "method",
    "reference",
    "delta_psnr",
    "delta_ssim",
    "delta_lpips",
    "delta_gs_num",
    "delta_train_time_s",
    "qcgi",
]

def build_comparison(ref_rows, reference):
    out = []
    for row in rows:
        ref = ref_rows[row["scene"]]
        comp = {
            "dataset": row["dataset"],
            "scene": row["scene"],
            "method": row["method"],
            "reference": reference,
        }
        for key in ("psnr", "ssim", "lpips", "gs_num", "train_time_s"):
            comp[f"delta_{key}"] = as_float(row, key) - as_float(ref, key)
        comp["qcgi"] = qcgi(row, ref)
        out.append(comp)
    return out

phase0_comp = build_comparison(baseline, "phase0_fastgs_big")
full_comp = build_comparison(full_i050, "descriptor_i050_full")
write_csv(combined / "comparison_vs_phase0.csv", phase0_comp, comparison_fields)
write_csv(combined / "comparison_vs_i050_full.csv", full_comp, comparison_fields)

summary = {
    "scene_count": len(rows),
    "missing_scenes": missing,
    "avg_psnr": avg([r["psnr"] for r in rows]),
    "avg_ssim": avg([r["ssim"] for r in rows]),
    "avg_lpips": avg([r["lpips"] for r in rows]),
    "avg_gs_num": avg([r["gs_num"] for r in rows]),
    "avg_train_time_s": avg([r["train_time_s"] for r in rows]),
    "avg_delta_psnr_phase0": avg([c["delta_psnr"] for c in phase0_comp]),
    "avg_delta_ssim_phase0": avg([c["delta_ssim"] for c in phase0_comp]),
    "avg_delta_lpips_phase0": avg([c["delta_lpips"] for c in phase0_comp]),
    "avg_delta_gs_num_phase0": avg([c["delta_gs_num"] for c in phase0_comp]),
    "avg_qcgi_phase0": avg([c["qcgi"] for c in phase0_comp]),
    "avg_delta_psnr_i050_full": avg([c["delta_psnr"] for c in full_comp]),
    "avg_delta_ssim_i050_full": avg([c["delta_ssim"] for c in full_comp]),
    "avg_delta_lpips_i050_full": avg([c["delta_lpips"] for c in full_comp]),
    "avg_delta_gs_num_i050_full": avg([c["delta_gs_num"] for c in full_comp]),
    "avg_qcgi_i050_full": avg([c["qcgi"] for c in full_comp]),
}
(combined / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if missing:
    raise SystemExit("Missing scenes: " + ", ".join(missing))
PY
}

start_sessions() {
  mkdir -p output/0010/debug_logs

  local wait_g0 wait_g1
  wait_g0="$(existing_pids_for mip_g0)"
  wait_g1="$(existing_pids_for mip_g1)"

  if ! tmux has-session -t "$SESSION_G0" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G0" \
      "cd '$ROOT_DIR' && export WAIT_PIDS='$wait_g0' && bash scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh worker-g0 2>&1 | tee output/0010/debug_logs/${SESSION_G0}.log"
  fi
  if ! tmux has-session -t "$SESSION_G1" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_G1" \
      "cd '$ROOT_DIR' && export WAIT_PIDS='$wait_g1' && bash scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh worker-g1 2>&1 | tee output/0010/debug_logs/${SESSION_G1}.log"
  fi
  if ! tmux has-session -t "$SESSION_MERGE" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_MERGE" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$SESSION_G0' 2>/dev/null || tmux has-session -t '$SESSION_G1' 2>/dev/null; do sleep 60; done; bash scripts/run_0010_descriptor_i050_until8000_pilot_tmux.sh merge 2>&1 | tee output/0010/debug_logs/${SESSION_MERGE}.log"
  fi

  tmux ls | grep '0010_i050_u8000' || true
}

case "${1:-}" in
  start)
    start_sessions
    ;;
  worker-g0)
    run_worker 0 "$BASE_OUT/mip_g0" bicycle garden stump
    ;;
  worker-g1)
    run_worker 1 "$BASE_OUT/mip_g1" counter kitchen bonsai
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
