# 0005 MipNeRF360 High-Gain PSNR Push 运行手册

## 环境

```bash
source .venv/bin/activate
```

## 评测策略

默认 final-only：

- 训练完整 30k。
- 只对最终 checkpoint 做 render / metrics。
- 不做每 2k checkpoint curve。

## Round 1 启动

GPU0 跑 5 个 outdoor / complex scenes：

```bash
tmux new-session -d -s 0005_tokenedge_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=0 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g0 --scenes bicycle flowers garden stump treehill --variant fastgs_big --config configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml --vfm-cache-template output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600 --resolution -1 --method-name tokenedge_vitl14_w1600_topk025_max --run-name tokenedge_vitl14_w1600_topk025_max_30k_final_r_auto > output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g0.log 2>&1"
```

GPU1 跑 4 个 indoor scenes：

```bash
tmux new-session -d -s 0005_tokenedge_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=1 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g1 --scenes room counter kitchen bonsai --variant fastgs_big --config configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml --vfm-cache-template output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600 --resolution -1 --method-name tokenedge_vitl14_w1600_topk025_max --run-name tokenedge_vitl14_w1600_topk025_max_30k_final_r_auto > output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g1.log 2>&1"
```

## 监控

```bash
tmux ls
nvidia-smi
tail -n 60 output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g0.log
tail -n 60 output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g1.log
```

## 结果整理

完成后合并：

- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g0/summary.csv`
- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g1/summary.csv`

并写回：

- `docs/experiments/0005_mipnerf360_high_gain_psnr_push/results.md`
- `docs/experiments/0005_mipnerf360_high_gain_psnr_push/review.md`
- `docs/roadmap.md`

## Round 2：RGB densify-until-21000

GPU0：

```bash
tmux new-session -d -s 0005_rgb_densify21_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=0 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g0 --scenes bicycle flowers garden stump treehill --variant fastgs_big --config configs/experiments/0005_fastgs_big_rgb_densify_until21000.yaml --resolution -1 --method-name fastgs_big_rgb_densify_until21000 --run-name fastgs_big_rgb_densify_until21000_30k_final_r_auto > output/0005/debug_logs/fastgs_big_rgb_densify_until21000_g0.log 2>&1"
```

GPU1：

```bash
tmux new-session -d -s 0005_rgb_densify21_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=1 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g1 --scenes room counter kitchen bonsai --variant fastgs_big --config configs/experiments/0005_fastgs_big_rgb_densify_until21000.yaml --resolution -1 --method-name fastgs_big_rgb_densify_until21000 --run-name fastgs_big_rgb_densify_until21000_30k_final_r_auto > output/0005/debug_logs/fastgs_big_rgb_densify_until21000_g1.log 2>&1"
```

监控：

```bash
tmux ls
tail -n 60 output/0005/debug_logs/fastgs_big_rgb_densify_until21000_g0.log
tail -n 60 output/0005/debug_logs/fastgs_big_rgb_densify_until21000_g1.log
```
