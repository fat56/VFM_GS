# 0005 MipNeRF360 High-Gain PSNR Push 结果

## 当前状态

Round 1 已 early-stop：DINOv2 ViT-L/14 token-edge 1.6K cache 在 4 个已完成场景上 PSNR 均值为 -0.0440，且平均多 195,124 个 Gaussians；虽然 SSIM/LPIPS 有正向，但已不可能达成 MipNeRF360 9 场景平均 PSNR +0.2 的当天目标。

Round 5 切到 soft late-prune：保持默认 15k densification，只把 18k/21k/24k/27k 后期 prune 的删除比例降到 50%。当前结果说明 prior、no-prune 和 60k 长训都不能作为全局默认。

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

- `output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g0/bicycle/logs/fastgs_big_rgb_densify_until21000_30k_final_r_auto/train.log`
- `output/0005/fastgs_big_rgb_densify_until21000/mipnerf360_g1/room/logs/fastgs_big_rgb_densify_until21000_30k_final_r_auto/train.log`

结果：失败，无有效 metrics。

失败点：`bicycle` 和 `room` 都在 iteration 18000 左右退出，报错为 `_RasterizeGaussiansBackward returned an invalid gradient at index 3 - got [0, 0, 3] but expected shape compatible with [0, 15, 3]`。

判定：这不是普通质量负例，而是延长 densification 触发了当前 rasterizer/SH feature 空张量路径的实现边界。为了继续当天 PSNR 冲刺，暂时不在此处修内核边界，先切到不改变 densify window 的 no-prune 容量上限探针。

## Round 3：FastGS big no-prune floor

配置：`configs/experiments/0005_fastgs_big_no_prune_floor.yaml`

输出：

- `output/0005/fastgs_big_no_prune_floor/mipnerf360_g0/summary.csv`
- `output/0005/fastgs_big_no_prune_floor/mipnerf360_g1/summary.csv`

本轮在 4 个场景后 early-stop。对照为同一 30k baseline。

| 场景 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2479 | -0.0168 | 0.7552 | -0.0003 | 0.2450 | +0.0003 | 2,570,509 | +1,012,429 |
| flowers | 21.6490 | +0.0277 | 0.6024 | +0.0002 | 0.3390 | -0.0017 | 1,958,298 | +823,466 |
| room | 32.3277 | +0.0270 | 0.9310 | +0.0003 | 0.1864 | -0.0016 | 1,281,650 | +711,460 |
| counter | 29.6154 | +0.0742 | 0.9183 | +0.0003 | 0.1753 | -0.0013 | 1,096,385 | +625,127 |
| **已完成均值** | 27.2100 | +0.0280 | 0.8015 | +0.0001 | 0.2364 | -0.0011 | 1,726,711 | +793,121 |

判定：停止继续跑剩余 5 场景。no-prune 保住大量容量，但 PSNR 只有薄正向，且 `bicycle` 退化；要达成 +0.2，剩余场景需要平均约 +0.315 PSNR，和已完成信号不匹配。

## Round 4：FastGS big 60k + LR 60k

配置：`configs/experiments/0005_fastgs_big_60k_lr60k.yaml`

输出：

- `output/0005/fastgs_big_60k_lr60k/mipnerf360_g0/summary.csv`
- `output/0005/fastgs_big_60k_lr60k/mipnerf360_g1/summary.csv`

本轮在 4 个场景后 early-stop。对照为同一 30k baseline。

| 场景 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.0702 | -0.1944 | 0.7470 | -0.0086 | 0.2470 | +0.0022 | 1,551,566 | -6,514 |
| flowers | 21.7803 | +0.1589 | 0.6119 | +0.0097 | 0.3360 | -0.0047 | 1,142,911 | +8,079 |
| room | 32.3645 | +0.0638 | 0.9302 | -0.0005 | 0.1871 | -0.0010 | 599,326 | +29,136 |
| counter | 29.6935 | +0.1523 | 0.9185 | +0.0005 | 0.1737 | -0.0028 | 498,919 | +27,661 |
| **已完成均值** | 27.2271 | +0.0452 | 0.8019 | +0.0003 | 0.2364 | -0.0016 | 948,181 | +14,591 |

判定：停止继续跑剩余 5 场景。60k 对 `flowers/counter` 很有帮助，但 `bicycle` 大幅退化，不能作为全局默认。

## Round 5：FastGS big soft-prune 0.50

配置：`configs/experiments/0005_fastgs_big_soft_prune050.yaml`

输出：

- `output/0005/fastgs_big_soft_prune050/mipnerf360_g0/summary.csv`
- `output/0005/fastgs_big_soft_prune050/mipnerf360_g1/summary.csv`

待完成后合并 9 场景，与同一 30k baseline 对齐。该轮只对最终 30k 做 render/metrics，不做 checkpoint curve。
