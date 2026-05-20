# 0009 Residual Orientation Prune-Protect 运行手册

## Smoke

先跑代码路径 smoke：

```bash
uv run --active python -m py_compile \
  src/vfm_gs/scorers/vfm_topology.py \
  src/vfm_gs/config/legacy_args.py
```

最小训练 smoke。这里用 CLI 覆盖 `vfm_active_from_iter=600`，确保第 600 step 的 densify scorer 实际执行新 backend：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0009_residual_orientation_protect_start24000_auto_topk005.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0009/residual_orientation_protect_smoke/treehill/smoke_700_r8 \
  --eval \
  --iterations 700 \
  --test_iterations 700 \
  --save_iterations 700 \
  --checkpoint_iterations 700 \
  --densification_interval 100 \
  --optimizer_type default \
  -r 8 \
  --vfm_cache_dir output/0002/vfm_cache/treehill_depth_anything_v2s_depth \
  --vfm_active_from_iter 600 \
  --dense 0.01 \
  --grad_abs_thresh 0.0018 \
  --quiet
CUDA_VISIBLE_DEVICES=0 uv run --active python -m vfm_gs.cli.render \
  -m output/0009/residual_orientation_protect_smoke/treehill/smoke_700_r8 \
  --iteration 700 \
  --skip_train \
  --quiet
CUDA_VISIBLE_DEVICES=0 uv run --active python -m vfm_gs.cli.metrics \
  -m output/0009/residual_orientation_protect_smoke/treehill/smoke_700_r8
```

## Round 1 Pilot

GPU0:

```bash
tmux new-session -d -s 0009_residual_protect_g0 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0009/residual_orientation_protect_pilot/mip_g0 --scenes room treehill --method-name residual_orientation_protect_start24000_auto_topk005 --run-name fastgs_big_30k_scene_override_r_auto --config configs/experiments/0009_residual_orientation_protect_start24000_auto_topk005.yaml --vfm-cache-template "output/0002/vfm_cache/{scene}_depth_anything_v2s_depth" --vfm-cache-feature depth > output/0009/debug_logs/residual_protect_g0.log 2>&1'
```

GPU1:

```bash
tmux new-session -d -s 0009_residual_protect_g1 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0009/residual_orientation_protect_pilot/mip_g1 --scenes stump --method-name residual_orientation_protect_start24000_auto_topk005 --run-name fastgs_big_30k_scene_override_r_auto --config configs/experiments/0009_residual_orientation_protect_start24000_auto_topk005.yaml --vfm-cache-template "output/0002/vfm_cache/{scene}_depth_anything_v2s_depth" --vfm-cache-feature depth > output/0009/debug_logs/residual_protect_g1.log 2>&1'
```

监控：

```bash
tmux ls
tail -n 80 output/0009/debug_logs/residual_protect_g0.log
tail -n 80 output/0009/debug_logs/residual_protect_g1.log
```

## 汇总

```bash
python scripts/build_0009_pilot_summary.py
```

若后续需要，补一个正式汇总脚本；Round 1 可先用 `summary.csv` 和既有 baseline 表手工记录。
