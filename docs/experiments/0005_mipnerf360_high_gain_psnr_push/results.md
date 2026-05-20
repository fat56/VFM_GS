# 0005 MipNeRF360 High-Gain PSNR Push 结果

## 当前状态

Round 1 准备启动：DINOv2 ViT-L/14 token-edge 1.6K cache，top-k25，`max` importance，FastGS big 1.6K 口径，全 9 场景 final-only。

## Round 1：token-edge ViT-L/1600 top-k25 max

配置：`configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml`

输出：

- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g0/summary.csv`
- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g1/summary.csv`

待完成后合并 9 场景，与 `output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 的 30k baseline 对齐。

| 场景 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| flowers | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| garden | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| stump | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| treehill | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| room | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| counter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| kitchen | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| bonsai | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **平均** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
