# 0006 Validation Selector + Online Depth Residual 运行手册

## 环境

```bash
source .venv/bin/activate
```

## Round 1：构建 selector 候选表

```bash
uv run --active python scripts/build_0006_selector_candidates.py --output-dir output/0006/validation_selector
```

输出：

- `output/0006/validation_selector/mipnerf360_depth_candidates.csv`
- `output/0006/validation_selector/cross_depth_candidates.csv`
- `output/0006/validation_selector/all_depth_candidates.csv`
- `output/0006/validation_selector/candidate_manifest.json`

## Round 1：双卡 train-split selector 评估

GPU0 跑 MipNeRF360：

```bash
tmux new-session -d -s 0006_selector_mip_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0006/debug_logs && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/evaluate_0001_train_selector.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0006/validation_selector/mipnerf360_train_selector --max-views 16 --view-stride 7 --resume > output/0006/debug_logs/selector_mip_g0.log 2>&1"
```

GPU1 跑 DB/Tandt：

```bash
tmux new-session -d -s 0006_selector_cross_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0006/debug_logs && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/evaluate_0001_train_selector.py --input-summary output/0006/validation_selector/cross_depth_candidates.csv --output-dir output/0006/validation_selector/cross_train_selector --max-views 16 --view-stride 7 --resume > output/0006/debug_logs/selector_cross_g1.log 2>&1"
```

监控：

```bash
tmux ls
tail -n 80 output/0006/debug_logs/selector_mip_g0.log
tail -n 80 output/0006/debug_logs/selector_cross_g1.log
nvidia-smi
```

## Round 1：结果整理

完成后查看：

- `output/0006/validation_selector/mipnerf360_train_selector/train_selector_averages.csv`
- `output/0006/validation_selector/cross_train_selector/train_selector_averages.csv`
- `output/0006/validation_selector/*/train_selector_recommendations.csv`

需要写回：

- `docs/experiments/0006_validation_selector_online_depth_residual/results.md`
- `docs/experiments/0006_validation_selector_online_depth_residual/review.md`

随后执行：

```bash
git status --short
git add docs/experiments/0006_validation_selector_online_depth_residual docs/experiments/index.md scripts/build_0006_selector_candidates.py
git commit -m "Add experiment 0006 selector plan"
git push
```

## Round 2：Online depth residual smoke

Round 2 第一部分只做 proxy smoke，不直接改 CUDA rasterizer：

- 用当前 Gaussian 中心投影加小半径 z-buffer splat，构造近似 rendered-depth proxy。
- 计算 Depth Anything cache 与 proxy depth / inverse-depth 的 residual。
- 对 `room kitchen bonsai stump counter` 比较 residual 分布、RGB error、已有 depth protect 选择的 candidate。
- 如果 residual 能区分 0004 负例和小正例，再进入 renderer alpha-weighted depth 输出实现。

GPU1 先跑 indoor / negative-heavy scenes：

```bash
tmux new-session -d -s 0006_depth_proxy_indoor_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0006/debug_logs && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/diagnose_0006_online_depth_residual_proxy.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0006/online_depth_residual_proxy/indoor_g1 --scenes room kitchen bonsai --max-views 8 --view-stride 17 --topk 0.10 --splat-radius 1 > output/0006/debug_logs/depth_proxy_indoor_g1.log 2>&1"
```

GPU0 等 `0006_selector_mip_g0` 结束后跑 small-positive / mixed scenes：

```bash
tmux new-session -d -s 0006_depth_proxy_mixed_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0006/debug_logs && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/diagnose_0006_online_depth_residual_proxy.py --input-summary output/0006/validation_selector/mipnerf360_depth_candidates.csv --output-dir output/0006/online_depth_residual_proxy/mixed_g0 --scenes stump counter --max-views 8 --view-stride 17 --topk 0.10 --splat-radius 1 > output/0006/debug_logs/depth_proxy_mixed_g0.log 2>&1"
```

监控：

```bash
tail -n 80 output/0006/debug_logs/depth_proxy_indoor_g1.log
tail -n 80 output/0006/debug_logs/depth_proxy_mixed_g0.log
```
