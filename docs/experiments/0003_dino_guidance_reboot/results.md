# 0003 实验结果

## 当前状态

0003 已完成 Phase 0 `bicycle` 诊断和 Phase 1 第一轮 high-res 620-step smoke。当前结论是：训练时同款 DINO descriptor residual 与 RGB 高误差区域只有弱重叠，单纯把 token grid 从 `10x16` 提高到 `75x114` 没有改善 overlap；但 `RGB broad candidate -> DINO rerank` 的训练链路已经打通，下一步可以进入 30k pilot。

## 0001 Token 粒度复核

| Cache | 图像源 | DINO 后端 | max_width | feature | 首图 shape | 每 token 大致覆盖 | 用途 |
|---|---|---|---:|---|---|---|---|
| `output/0001/vfm_cache/bicycle_dinov2_vits14` | `images_8` | ViT-S/14 | 224 | patch tokens | `10x16x384` | `-r 8` 下约 `39x41` px；high-res 复用时约 `100x106` px | 0001 descriptor 主线 |
| `output/0001/vfm_cache/bicycle_dinov2_vits14_w518` | `images_8` | ViT-S/14 | 518 | patch tokens | `24x37x384` | `-r 8` 下约 `17x17` px | 早期高分辨率 cache 探测，未成主线 |
| `output/0001/vfm_cache_large/bicycle_dinov2_vitl14_token_edge_w1600` | `images` | ViT-L/14 | 1600 | token edge | `75x114` | high-res 下约 `14x14` px | high-res token-edge 线，不是 descriptor residual |

事实结论：0001 descriptor 主线使用的 DINO map 非常粗，尤其在 high-res 复验中复用 `10x16` token grid 时，metric map 的空间定位能力不足。

## 0002 Overlap 证据

来自 `scripts/diagnose_prior_overlap.py` 的 high-res `bicycle` 诊断：

| Prior | top-k | prior/RGB IoU | prior/RGB recall | 备注 |
|---|---:|---:|---:|---|
| Depth Anything relative depth | 25% | 0.226 | 0.367 | 比较接近 RGB 瓶颈，但仍有限 |
| Depth Anything relative depth | 10% | 0.124 | 0.219 | 最难区域重叠较低 |
| Depth Anything depth edge | 25% | 0.175 | 0.297 | 较弱 |
| DINO ViT-L token edge | 25% | 0.149 | 0.259 | 更像结构重要性，不是 RGB error proxy |
| DINO ViT-L token edge | 10% | 0.068 | 0.127 | 与 RGB 最大误差区域明显错位 |

解释边界：这里的 DINO 是 token-edge prior，不是 0001 训练时的 descriptor residual。0003 必须补做真实 descriptor residual overlap，不能只用 token-edge 诊断替代，也不能用这组 IoU 直接解释 0001 descriptor 为什么指标涨得薄。

## 待产出

| 日期 | 阶段 | 场景 | 内容 | 输出 | 结论 |
|---|---|---|---|---|---|
| 2026-05-12 | Phase 0 | bicycle | 训练时同款 render-vs-GT DINO cosine error overlap | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w{224,518,1600}_topk{25,10}` | 全局 top-k overlap 只比随机基线略高，descriptor residual 不是可靠 RGB error proxy |
| 2026-05-12 | Phase 0 | bicycle | 224/518/1600 token 粒度比较 | `scripts/diagnose_dino_descriptor_residual.py` | token 变细没有改善；w1600 的 IoU 和 Spearman 反而最低 |
| 2026-05-12 | Phase 1 | bicycle | RGB-broad candidate + DINO rerank + late activation 620-step smoke | `output/0003/{rgb_broad_bicycle_620_r_auto,dino_rgb_rerank_l025_bicycle_620_r_auto}` | 链路健康；620-step 只作集成验证，不作质量结论 |
| TBD | Phase 2 | bicycle/stump/treehill/bonsai | start_iter/lambda/broad top-k 30k pilot | TBD | TBD |
| TBD | Phase 3 | TBD | DINO prune-protect pilot | TBD | 只在 rerank 成立后推进 |

## 2026-05-12 Phase 0：训练时同款 DINO Descriptor Residual 诊断

新增 `scripts/diagnose_dino_descriptor_residual.py`，复现 `dinov2_descriptor_cosine` 训练后端的中间 residual map：

```text
baseline render image -> DINO patch tokens
GT/source image cache  -> DINO patch tokens
residual = 0.5 * clamp(1 - cosine(render_token, gt_token), 0, 2)
```

诊断对象为 high-res `bicycle` matched baseline：`output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto`。`w224` 和 `w518` 直接复用 `output/0001/vfm_cache/bicycle_dinov2_vits14*`；`w1600` 为避免先生成全量 cache，诊断脚本对 render/GT 即时提取 ViT-S/14 tokens。三组均使用 `smooth_kernel=3` 和 bilinear upsample，以贴近 0001 descriptor 主线。

| Scale | top-k | grid | IoU | random IoU | lift | recall | spearman pixel | DINO top-k in RGB top-50 | RGB L1 in DINO top-k | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| w224 | 25% | 10x16 | 0.1638 | 0.1429 | 1.15x | 0.2794 | 0.0422 | 0.5227 | 0.04324 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w224_topk25` |
| w518 | 25% | 24x37 | 0.1615 | 0.1429 | 1.13x | 0.2771 | 0.0626 | 0.5332 | 0.04246 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w518_topk25` |
| w1600 | 25% | 75x114 | 0.1554 | 0.1429 | 1.09x | 0.2680 | 0.0267 | 0.5222 | 0.04277 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w1600_topk25` |
| w224 | 10% | 10x16 | 0.0713 | 0.0526 | 1.35x | 0.1316 | 0.0422 | 0.5324 | 0.04437 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w224_topk10` |
| w518 | 10% | 24x37 | 0.0663 | 0.0526 | 1.26x | 0.1234 | 0.0626 | 0.5503 | 0.04432 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w518_topk10` |
| w1600 | 10% | 75x114 | 0.0637 | 0.0526 | 1.21x | 0.1189 | 0.0267 | 0.5403 | 0.04491 | `output/0003/diagnostics/bicycle_dino_descriptor_residual_w1600_topk10` |

判断：

- 真实 descriptor residual 比 0002 的 DINO token-edge 略好，但仍接近随机 overlap。top-25% 随机 IoU 为 0.1429，descriptor 只有 0.155~0.164；top-10% 随机 IoU 为 0.0526，descriptor 只有 0.064~0.071。
- 增大 token 粒度没有解决错位。`w1600` 的 top-25% IoU 为 0.1554，低于 `w224/w518`；pixel Spearman 也只有 0.0267。
- DINO top-k 内的 RGB L1 确实高于全图均值，例如 top-25% `w224` 为 0.04324 vs 全图 0.03937，但提升很小，说明它偏向“稍难区域”，不是 RGB loss 主要瓶颈。
- DINO top-k 约 52%~55% 落入 RGB top-50% broad candidate。这个结果支持下一步做 `RGB broad candidate -> DINO rerank`，但不支持 DINO 单独生成 densification 区域。

## 决策

0003 不继承 0001 的“DINO descriptor 已经定位清楚”作为前提。0001 的全图质量正向仍是有价值证据，但 Phase 0 已确认：在 high-res `bicycle` 上，训练时同款 DINO residual 的全局 top-k 与 RGB 高误差区域只有弱重叠。第一训练候选改为 RGB 放宽候选内部的 DINO rerank，而不是裸 DINO top-k 或 DINO 主动扩候选。

## 2026-05-12 Phase 1：RGB Broad + DINO Rerank 620-step Smoke

代码新增两个 densification importance mode：

- `rgb_broad`：使用 RGB/FastGS importance 作为候选，配合较宽松的 `loss_thresh=0.05` 和 `vfm_rgb_broad_topk=0.50`，作为 matched control。
- `rgb_rerank`：仍由 RGB/FastGS 生成 broad candidate，DINO descriptor residual 只在候选内部通过 `RGB_importance * (1 + lambda * normalize(DINO))` 调整强度，不允许 DINO 单独拉入 RGB 低误差区域。

新增配置：

- `configs/experiments/0003_dino_descriptor_rgb_broad.yaml`
- `configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml`

本轮在 high-res `bicycle`、原图输入/1.6K 自动缩放口径下跑 620 iteration。`DINO rerank` smoke 通过 `--vfm_active_from_iter 600` 只在最后一次 densification 介入，用于确认 late activation、DINO cache、在线 render-token extraction 和 Gaussian score 回写链路正常。

| Run | 配置 | DINO 介入 | PSNR | SSIM | LPIPS | Gaussians | 输出 |
|---|---|---:|---:|---:|---:|---:|---|
| RGB broad control 620 | `0003_dino_descriptor_rgb_broad.yaml` | active from 0 | 19.4699 | 0.4046 | 0.6282 | 63,439 | `output/0003/rgb_broad_bicycle_620_r_auto` |
| DINO RGB rerank l0.25 620 | `0003_dino_descriptor_rgb_rerank_l025.yaml` | active from 600 | 19.4483 | 0.4051 | 0.6281 | 63,442 | `output/0003/dino_rgb_rerank_l025_bicycle_620_r_auto` |

日志：

- RGB broad train/render/metrics：`output/0003/logs/rgb_broad_bicycle_620_r_auto.{train,render,metrics}.log`
- DINO rerank train/render/metrics：`output/0003/logs/dino_rgb_rerank_l025_bicycle_620_r_auto.{train,render,metrics}.log`

判断：

- 两组 train/render/metrics 均完成，说明新增 mode、配置解析、cache preflight、DINOv2 repo 加载和 1.6K render/metrics 链路健康。
- DINO rerank 在 600 iter 触发 DINOv2 token extraction，训练没有报错；点数与 RGB broad control 几乎一致，符合“只在候选内部 rerank”的预期。
- 620-step 指标差异非常小，且训练过短，不支持质量判断。Phase 2 必须跑 30k matched pilot，第一组固定 `broad_topk=0.50`、`lambda=0.25`，扫描 `DINO_start_iter=7000/9000/11000`，并与 RGB broad 30k matched control 对照。

## 2026-05-12 Phase 2：Bicycle 30k 首轮 Matched Pilot

本轮使用双卡同时训练：GPU0 跑 `RGB broad` matched control，GPU1 跑 `DINO RGB rerank l0.25, start_iter=9000`。服务器未安装 `screen`，实际使用 `setsid` wrapper 脱离当前 SSH；两组 train/render/metrics 均完成。训练仍保持 high-res 原图输入/1.6K 自动缩放口径。

| Run | 配置 | active_from | PSNR | SSIM | LPIPS | Gaussians | Train wall | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RGB broad control 30k | `0003_dino_descriptor_rgb_broad.yaml` | 0 | 25.3627 | 0.7656 | 0.2273 | 1,883,915 | 243s | `output/0003/rgb_broad_bicycle_30k_r_auto` |
| DINO RGB rerank l0.25 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` | 9000 | 25.3538 | 0.7659 | 0.2260 | 1,915,967 | 242s | `output/0003/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto` |

相对 RGB broad control：

| Run | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussians | 判断 |
|---|---:|---:|---:|---:|---|
| DINO RGB rerank l0.25 start9000 | -0.0088 | +0.0003 | -0.0013 | +32,052 | 弱混合信号：感知指标小幅正向，但 PSNR 小负且点数更多 |

参考 5090 high-res FastGS big `bicycle` baseline `25.2569 / 0.7553 / 0.2450`、1,560,209 点：RGB broad 和 DINO rerank 都能明显提升 SSIM/LPIPS，但主要代价是 0.32M~0.36M 级别的额外 Gaussians。当前不能把这解释为 DINO rerank 的独立收益；更可能是放宽 RGB 候选本身带来的容量和质量提升。

日志：

- RGB broad：`output/0003/logs/rgb_broad_bicycle_30k_r_auto.{train,render,metrics}.log`
- DINO rerank：`output/0003/logs/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto.{train,render,metrics}.log`

判断：

- Phase 2 首轮证明 `RGB broad -> DINO rerank` 30k 链路健康，且后期介入 `start_iter=9000` 没有训练稳定性问题。
- DINO rerank 相对 matched RGB broad control 没有形成明确全图质量收益。SSIM/LPIPS 的微弱正向可能仍值得继续看 start_iter/lambda，但不能直接扩全场景。
- 下一轮先用双卡扫描 `start_iter=7000/11000`。如果仍是 PSNR 负、点数增且感知指标只微正，应转向 `lambda=0.10`、显式 final top-m 或局部指标诊断，而不是继续加大 DINO 影响。
