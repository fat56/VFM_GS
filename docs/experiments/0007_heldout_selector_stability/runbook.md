# 0007 Held-out Selector Stability 运行手册

## Round 1：smoke

先用 `bonsai` 两个方法确认脚本能正确输出 selector / holdout 两套指标：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/evaluate_0007_heldout_train_selector.py \
  --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv \
  --output-dir output/0007/heldout_selector_smoke \
  --scenes bonsai \
  --methods baseline depth_auto_topk \
  --max-views 4 \
  --view-stride 7 \
  --resume
```

## Round 1：双卡全量评估

GPU0 跑 MipNeRF360：

```bash
tmux new-session -d -s 0007_heldout_mip_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0007/debug_logs && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/evaluate_0007_heldout_train_selector.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0007/heldout_selector/mipnerf360 --max-views 16 --view-stride 7 --resume > output/0007/debug_logs/heldout_mip_g0.log 2>&1"
```

GPU1 跑 DB/Tandt：

```bash
tmux new-session -d -s 0007_heldout_cross_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0007/debug_logs && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/evaluate_0007_heldout_train_selector.py --input-summary output/0006/validation_selector/cross_depth_candidates.csv --output-dir output/0007/heldout_selector/cross --max-views 16 --view-stride 7 --resume > output/0007/debug_logs/heldout_cross_g1.log 2>&1"
```

## 监控

```bash
tmux ls
tail -n 80 output/0007/debug_logs/heldout_mip_g0.log
tail -n 80 output/0007/debug_logs/heldout_cross_g1.log
```

## 输出

每个分组都会生成：

- `heldout_candidate_metrics.csv`
- `heldout_selector_recommendations.csv`
- `heldout_selector_averages.csv`
- `heldout_baseline_averages.csv`

## 提交

实验设置提交：

```bash
git add docs/experiments/0007_heldout_selector_stability docs/experiments/index.md scripts/evaluate_0007_heldout_train_selector.py
git commit -m "Add experiment 0007 heldout selector"
git push
```

结果整理后再单独提交：

```bash
git add docs/experiments/0007_heldout_selector_stability
git commit -m "Record experiment 0007 heldout selector results"
git push
```
