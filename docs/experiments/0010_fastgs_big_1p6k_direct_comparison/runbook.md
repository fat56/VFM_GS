# 0010 Runbook

## Descriptor i0.50, legacy 0001 cache

Detached tmux launcher:

```bash
bash scripts/run_0010_descriptor_i050_tmux.sh start
```

Sessions:

```bash
tmux attach -t 0010_i050_g0
tmux attach -t 0010_i050_g1
tmux attach -t 0010_i050_merge
```

Logs:

```text
output/0010/debug_logs/0010_i050_g0.log
output/0010/debug_logs/0010_i050_g1.log
output/0010/debug_logs/0010_i050_merge.log
```

The launcher only reruns scenes missing `results.json`. If an interrupted scene directory exists without metrics, it is moved under `output/0010/debug_artifacts/` before rerun.

GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0010/descriptor_i050_fastgs_big_legacy_cache/mip_g0 \
  --scenes bicycle flowers garden stump treehill \
  --train-images images \
  --iterations 30000 \
  --resolution -1 \
  --variant fastgs_big \
  --densification-interval 100 \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  --method-name descriptor_i050_fastgs_big_legacy_cache \
  --run-name descriptor_i050_fastgs_big_legacy_cache_30k_r_auto \
  --vfm-cache-template 'output/0001/vfm_cache/{scene}_dinov2_vits14'
```

GPU1:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0010/descriptor_i050_fastgs_big_legacy_cache/mip_g1 \
  --scenes room counter kitchen bonsai \
  --train-images images \
  --iterations 30000 \
  --resolution -1 \
  --variant fastgs_big \
  --densification-interval 100 \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  --method-name descriptor_i050_fastgs_big_legacy_cache \
  --run-name descriptor_i050_fastgs_big_legacy_cache_30k_r_auto \
  --vfm-cache-template 'output/0001/vfm_cache/{scene}_dinov2_vits14'
```

## Baseline

Use:

```text
output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv
```
