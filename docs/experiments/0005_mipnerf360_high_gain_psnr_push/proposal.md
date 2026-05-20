# 0005 MipNeRF360 High-Gain PSNR Push

## 核心假设

0004 的 late prune-side auxiliary 更像减伤器，难以冲到 MipNeRF360 9 场景平均 PSNR +0.2。0005 改成高增益路线：优先使用已有正向证据最强的 DINO token-edge / descriptor densification 引导，在 FastGS big 1.6K 口径下直接跑全 9 场景，目标是尽快找到平均 PSNR 至少超过 baseline +0.2 的配置。

## 变体 / 配置

- 变体：`fastgs_big`
- 第一轮配置：`configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml`
- 打分器：`vfm_topology_scorer`
- prior：DINOv2 ViT-L/14 token-edge，1.6K cache，top-k25，`max` importance
- baseline：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 的 30k FastGS big 结果

## 运行命令

GPU0：

```bash
tmux new-session -d -s 0005_tokenedge_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=0 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g0 --scenes bicycle flowers garden stump treehill --variant fastgs_big --config configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml --vfm-cache-template output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600 --resolution -1 --method-name tokenedge_vitl14_w1600_topk025_max --run-name tokenedge_vitl14_w1600_topk025_max_30k_final_r_auto > output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g0.log 2>&1"
```

GPU1：

```bash
tmux new-session -d -s 0005_tokenedge_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0005/debug_logs && CUDA_VISIBLE_DEVICES=1 python scripts/run_0001_fastgs_big_eval.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g1 --scenes room counter kitchen bonsai --variant fastgs_big --config configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml --vfm-cache-template output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600 --resolution -1 --method-name tokenedge_vitl14_w1600_topk025_max --run-name tokenedge_vitl14_w1600_topk025_max_30k_final_r_auto > output/0005/debug_logs/tokenedge_vitl14_w1600_topk025_max_g1.log 2>&1"
```

## 数据集

- 数据集：MipNeRF360
- 场景：9 scenes
- 分辨率：原图输入，FastGS 1.6K 自动缩放，`-r -1`
- 评测：final-only，30k 后 render / metrics

## 指标

| 指标 | 基线 | 实验 | 差值 |
|---|---:|---:|---:|
| PSNR | FastGS big 30k | TBD | 目标 >= +0.2000 |
| SSIM | FastGS big 30k | TBD | TBD |
| LPIPS | FastGS big 30k | TBD | TBD |
| Gaussian 数量 | FastGS big 30k | TBD | TBD |

## 失败记录

TBD

## 决策

若第一轮不能达到 +0.2，下一轮优先尝试 DINO descriptor top-k25 `max` 或更强的 token-edge / descriptor 混合增长，而不是继续 0004 fixed prune-protect。

## 下一步

启动双卡 tmux 跑第一轮全 9 场景。
