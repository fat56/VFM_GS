#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

START_TS=$(date +%s)
echo "[large-res-full] start $(date -Is)"

COMMON_ARGS=(
  --train-images images
  --cache-images images
  --resolution -1
  --cache-root output/0001/vfm_cache_large
  --cache-max-width 1600
  --cache-storage npz_uint8
  --project-token-edge
  --dino-backend dinov2_vitl14
  --dinov2-repo output/0001/external/dinov2
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050.yaml
  --method-name large_res_dinov2_vitl14_token_edge_weighted_i050
  --run-name vfm_dinov2_vitl14_token_edge_weighted_i050_30k_r_auto
)

uv run --active python scripts/run_0001_dino_weighted_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0001/large_res_vitl_full/mipnerf360 \
  --scenes bicycle bonsai counter flowers garden kitchen room stump treehill \
  "${COMMON_ARGS[@]}"

echo "[large-res-full] mipnerf360 done $(date -Is)"

uv run --active python scripts/run_0001_dino_weighted_eval.py \
  --dataset-name db \
  --dataset-root datasets/tandt_db/db \
  --output-root output/0001/large_res_vitl_full/db \
  --scenes drjohnson playroom \
  "${COMMON_ARGS[@]}"

echo "[large-res-full] db done $(date -Is)"

uv run --active python scripts/run_0001_dino_weighted_eval.py \
  --dataset-name tandt \
  --dataset-root datasets/tandt_db/tandt \
  --output-root output/0001/large_res_vitl_full/tandt \
  --scenes train truck \
  "${COMMON_ARGS[@]}"

END_TS=$(date +%s)
echo "[large-res-full] tandt done $(date -Is)"
echo "[large-res-full] elapsed_seconds=$((END_TS - START_TS))"
