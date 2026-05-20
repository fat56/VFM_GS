# 0008 Residual Orientation Gating 运行手册

## Round 1：summary smoke

复用 0006 的 residual proxy 输出，先检查汇总脚本：

```bash
uv run --active python scripts/summarize_0008_residual_orientation_gating.py \
  --inputs \
    output/0006/online_depth_residual_proxy/indoor_g1/per_view.csv \
    output/0006/online_depth_residual_proxy/mixed_g0/per_view.csv \
  --output-dir output/0008/residual_orientation_gating_smoke
```

## Round 2：双卡 full MipNeRF360 proxy

GPU0 跑 outdoor / large scenes：

```bash
tmux new-session -d -s 0008_residual_proxy_mip_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0008/debug_logs && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/diagnose_0006_online_depth_residual_proxy.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0008/residual_proxy/mip_g0 --scenes bicycle flowers garden treehill --max-views 8 --view-stride 17 --topk 0.10 --splat-radius 1 > output/0008/debug_logs/residual_proxy_mip_g0.log 2>&1"
```

GPU1 跑 indoor / mixed scenes：

```bash
tmux new-session -d -s 0008_residual_proxy_mip_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0008/debug_logs && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/diagnose_0006_online_depth_residual_proxy.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0008/residual_proxy/mip_g1 --scenes bonsai counter kitchen room stump --max-views 8 --view-stride 17 --topk 0.10 --splat-radius 1 > output/0008/debug_logs/residual_proxy_mip_g1.log 2>&1"
```

## Round 2：orientation 汇总

```bash
uv run --active python scripts/summarize_0008_residual_orientation_gating.py \
  --inputs \
    output/0008/residual_proxy/mip_g0/per_view.csv \
    output/0008/residual_proxy/mip_g1/per_view.csv \
  --output-dir output/0008/residual_orientation_gating_mipnerf360
```

## 监控

```bash
tmux ls
tail -n 80 output/0008/debug_logs/residual_proxy_mip_g0.log
tail -n 80 output/0008/debug_logs/residual_proxy_mip_g1.log
```

## 输出

- `per_view_orientation.csv`
- `scene_orientation_summary.csv`
- `dataset_orientation_summary.csv`
- `overall_orientation_summary.json`

## 提交

实验设置提交：

```bash
git add docs/experiments/0008_residual_orientation_gating docs/experiments/index.md scripts/summarize_0008_residual_orientation_gating.py
git commit -m "Add experiment 0008 residual orientation gating"
git push
```

结果整理后再单独提交：

```bash
git add docs/experiments/0008_residual_orientation_gating docs/experiments/index.md
git commit -m "Record experiment 0008 residual orientation results"
git push
```
