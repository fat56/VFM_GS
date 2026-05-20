# 0005 MipNeRF360 High-Gain PSNR Push 结果

## 当前状态

Round 1 已 early-stop：DINOv2 ViT-L/14 token-edge 1.6K cache 在 4 个已完成场景上 PSNR 均值为 -0.0440，且平均多 195,124 个 Gaussians；虽然 SSIM/LPIPS 有正向，但已不可能达成 MipNeRF360 9 场景平均 PSNR +0.2 的当天目标。

Round 2 切到质量优先 FastGS/RGB 自身改动：`densify_until_iter=21000`，不引入 VFM，验证延长后期增长是否能补足默认 15k 之后的边缘/细节容量。

## Round 1：token-edge ViT-L/1600 top-k25 max

配置：`configs/experiments/0005_dinov2_token_edge_vitl14_w1600_topk025_max_fastgs_big.yaml`

输出：

- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g0/summary.csv`
- `output/0005/tokenedge_vitl14_w1600_topk025_max_fastgs_big/mipnerf360_g1/summary.csv`

本轮在 4 个场景后 early-stop。对照为 `output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 的 30k baseline。

| 场景 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.3451 | +0.0805 | 0.7644 | +0.0088 | 0.2297 | -0.0151 | 1,794,991 | +236,911 |
| flowers | 21.5948 | -0.0266 | 0.6039 | +0.0017 | 0.3369 | -0.0038 | 1,352,864 | +218,032 |
| garden | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| stump | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| treehill | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| room | 32.0677 | -0.2330 | 0.9329 | +0.0022 | 0.1784 | -0.0097 | 739,277 | +169,087 |
| counter | 29.5443 | +0.0032 | 0.9201 | +0.0022 | 0.1704 | -0.0061 | 627,722 | +156,464 |
| kitchen | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| bonsai | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **已完成均值** | 27.1380 | -0.0440 | 0.8053 | +0.0037 | 0.2289 | -0.0087 | 1,128,714 | +195,124 |

判定：停止继续跑剩余 5 场景。若剩余场景要把 9 场景均值拉到 +0.2，需要平均接近 +0.40 PSNR，同一 token-edge 信号已没有足够可信空间。

## Round 2：FastGS big RGB densify-until-21000

配置：`configs/experiments/0005_fastgs_big_rgb_densify_until21000.yaml`

输出：

- `output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g0/summary.csv`
- `output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g1/summary.csv`

待完成后合并 9 场景，与同一 30k baseline 对齐。该轮只在最终 30k 做 render/metrics，不再做 checkpoint curve。
