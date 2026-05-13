# 0003 实验结果

## 当前状态

0003 已完成 Phase 0 `bicycle` 诊断、Phase 1 high-res 620-step smoke，以及 Phase 2 high-res `bicycle` 30k matched pilot、局部区域诊断和 final-topm 容量锁定判别。当前结论是：训练时同款 DINO descriptor residual 与 RGB 高误差区域只有弱重叠，单纯把 token grid 从 `10x16` 提高到 `75x114` 没有改善 overlap；`RGB broad candidate -> DINO rerank` 的训练链路健康，但未证明 DINO selector 的独立价值。final-topm 能把点数压回接近 RGB broad control，但全图收益仍很薄，DINO-only 区域继续退化，因此 0003 的 DINO rerank 训练分支暂时收束。Phase 3 改为只测试 DINO prune-protect：DINO 不主动删除、不参与 densification，只在 RGB pruning high-score 候选中降低误删概率。

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
| 2026-05-12 | Phase 2 | bicycle | `lambda=0.25` start_iter 7000/9000/11000 30k matched pilot | `output/0003/dino_rgb_rerank_l025_start{7000,9000,11000}_bicycle_30k_r_auto` | start7000 三项质量小幅正向但点数更多；9000/11000 为 PSNR 小负、SSIM/LPIPS 微正 |
| 2026-05-12 | Phase 2 | bicycle | `lambda=0.10` start_iter 7000/9000 30k matched pilot | `output/0003/dino_rgb_rerank_l010_start{7000,9000}_bicycle_30k_r_auto` | 省点约 10k，但 PSNR 低于 RGB broad control，未优于 l0.25 |
| 2026-05-12 | Phase 2 | bicycle | 局部区域诊断：RGB broad fixed mask 下比较 RGB/DINO/DINO-only/intersection 区域 | `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010` | DINO-only top25 L1 上升、PSNR 下降；收益主要来自 RGB 高误差候选轨迹 |
| 2026-05-12 | Phase 2 | bicycle | final-topm 容量锁定实现、620-step smoke 和 30k 判别 | `output/0003/dino_rgb_rerank_finaltopm_l{025,010}_*`；`output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_finaltopm` | 容量锁定有效，但全图收益很薄，DINO-only top25 仍退化 |
| 2026-05-13 | Phase 3 | bicycle | DINO prune-protect only 设计与配置 | `configs/experiments/0003_dino_descriptor_prune_protect_only.yaml` | 待跑；DINO 只保护 RGB high-prune candidate，不参与 densify 或主动 pruning |

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

## 2026-05-12 Phase 2：Bicycle 30k Start-Iter Matched Pilot

本轮使用双卡分两批训练：第一批 GPU0 跑 `RGB broad` matched control，GPU1 跑 `DINO RGB rerank l0.25, start_iter=9000`；第二批 GPU0/GPU1 并行扫描 `start_iter=7000/11000`。服务器未安装 `screen`，实际使用 `setsid` wrapper 脱离当前 SSH；所有 train/render/metrics 均完成。训练仍保持 high-res 原图输入/1.6K 自动缩放口径。

| Run | 配置 | active_from | PSNR | SSIM | LPIPS | Gaussians | Train wall | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RGB broad control 30k | `0003_dino_descriptor_rgb_broad.yaml` | 0 | 25.3627 | 0.7656 | 0.2273 | 1,883,915 | 243s | `output/0003/rgb_broad_bicycle_30k_r_auto` |
| DINO RGB rerank l0.25 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` | 7000 | 25.3695 | 0.7663 | 0.2255 | 1,924,629 | 245s | `output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto` |
| DINO RGB rerank l0.25 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` | 9000 | 25.3538 | 0.7659 | 0.2260 | 1,915,967 | 242s | `output/0003/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto` |
| DINO RGB rerank l0.25 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` | 11000 | 25.3515 | 0.7660 | 0.2262 | 1,905,234 | 241s | `output/0003/dino_rgb_rerank_l025_start11000_bicycle_30k_r_auto` |

相对 RGB broad control：

| Run | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussians | 判断 |
|---|---:|---:|---:|---:|---|
| DINO RGB rerank l0.25 start7000 | +0.0068 | +0.0007 | -0.0018 | +40,714 | 唯一三项质量均小幅正向的点，但容量代价更高，收益仍很薄 |
| DINO RGB rerank l0.25 start9000 | -0.0088 | +0.0003 | -0.0013 | +32,052 | 弱混合信号：感知指标小幅正向，但 PSNR 小负且点数更多 |
| DINO RGB rerank l0.25 start11000 | -0.0112 | +0.0004 | -0.0011 | +21,319 | 介入更晚可减少点数增量，但质量收益没有扩大 |

参考 5090 high-res FastGS big `bicycle` baseline `25.2569 / 0.7553 / 0.2450`、1,560,209 点：RGB broad 和 DINO rerank 都能明显提升 SSIM/LPIPS，但主要代价是 0.32M~0.36M 级别的额外 Gaussians。当前不能把这解释为 DINO rerank 的独立收益；更可能是放宽 RGB 候选本身带来的容量和质量提升。

日志：

- RGB broad：`output/0003/logs/rgb_broad_bicycle_30k_r_auto.{train,render,metrics}.log`
- DINO rerank start7000：`output/0003/logs/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto.{train,render,metrics}.log`
- DINO rerank start9000：`output/0003/logs/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto.{train,render,metrics}.log`
- DINO rerank start11000：`output/0003/logs/dino_rgb_rerank_l025_start11000_bicycle_30k_r_auto.{train,render,metrics}.log`

判断：

- Phase 2 证明 `RGB broad -> DINO rerank` 30k 链路健康，`start_iter=7000/9000/11000` 都没有训练稳定性问题。
- `start7000` 是当前最好的 DINO rerank 点，但相对 RGB broad control 只多 +0.0068 PSNR、+0.0007 SSIM、LPIPS -0.0018，同时多 40,714 个 Gaussians；这更像弱增益，而不是足以扩全场景的明确收益。
- `start9000/11000` 则维持 PSNR 小负、SSIM/LPIPS 微正。下一轮应优先降低 DINO rerank 强度到 `lambda=0.10`，或引入显式 final top-m/局部指标诊断，而不是继续加大 DINO 影响。

## 2026-05-12 Phase 2：Bicycle 30k Lambda 0.10 Matched Pilot

本轮按上一轮结论降低 DINO rerank 强度，使用双卡并行训练 `lambda=0.10, start_iter=7000/9000`。两组 train/render/metrics 均完成，仍保持 high-res 原图输入/1.6K 自动缩放口径。

| Run | 配置 | lambda | active_from | PSNR | SSIM | LPIPS | Gaussians | Train wall | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DINO RGB rerank l0.10 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` + CLI override | 0.10 | 7000 | 25.3519 | 0.7660 | 0.2262 | 1,913,988 | 244s | `output/0003/dino_rgb_rerank_l010_start7000_bicycle_30k_r_auto` |
| DINO RGB rerank l0.10 30k | `0003_dino_descriptor_rgb_rerank_l025.yaml` + CLI override | 0.10 | 9000 | 25.3556 | 0.7660 | 0.2261 | 1,907,193 | 242s | `output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto` |

相对 RGB broad control：

| Run | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussians | 相对 l0.25 同 start | 判断 |
|---|---:|---:|---:|---:|---|---|
| l0.10 start7000 | -0.0108 | +0.0004 | -0.0011 | +30,073 | 少 10,641 点，但 PSNR -0.0176、SSIM/LPIPS 也回落 | 降低 lambda 破坏了 start7000 的唯一三项小正向 |
| l0.10 start9000 | -0.0070 | +0.0004 | -0.0013 | +23,278 | 少 8,774 点，PSNR +0.0018、LPIPS 基本持平 | 与 l0.25 start9000 近似，仍不是明确收益 |

日志：

- DINO rerank l0.10 start7000：`output/0003/logs/dino_rgb_rerank_l010_start7000_bicycle_30k_r_auto.{train,render,metrics,eval}.log`
- DINO rerank l0.10 start9000：`output/0003/logs/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto.{train,render,metrics,eval}.log`

判断：

- `lambda=0.10` 确实略微减少 Gaussians，但只减少 8k~11k，不足以改变 RGB broad + DINO rerank 的容量结论。
- 低 lambda 没有保住 `lambda=0.25 start7000` 的 PSNR 小正向；start9000 与 l0.25 基本同档。
- 0003 当前不应扩多场景。本轮判断已推动后续 final-topm 和局部区域诊断，用来确认 DINO rerank 是否真的改善 DINO/RGB 交集区域，而不是继续扫相近 lambda/start_iter。

## 2026-05-12 Phase 2：局部区域诊断

新增 `scripts/diagnose_0003_local_regions.py`，用 RGB broad control 的 render-vs-GT DINO residual 和 RGB L1 构造固定 mask，再比较不同 run 在这些区域里的 RGB L1。当前只报告空间 L1/PSNR；LPIPS 现有实现只返回全图标量，暂不伪造空间 LPIPS map。

诊断命令输出：

- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010/summary.json`
- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010/per_view.csv`

mask 仍显示 DINO/RGB 错位：top-25% DINO/RGB IoU 为 0.1627，DINO-only top-25% 占 7,654,679 像素，RGB-only top-25% 也占 7,654,679 像素；top-10% IoU 为 0.0716。

相对 RGB broad control 的局部 L1 变化：

| Region, top25 masks from RGB broad | l0.25 start7000 ΔL1 | l0.25 start7000 ΔPSNR | l0.10 start9000 ΔL1 | l0.10 start9000 ΔPSNR | 解释 |
|---|---:|---:|---:|---:|---|
| all | -0.000028 | -0.0234 | -0.000001 | -0.0341 | 全图变化接近 0，MSE/PSNR 反而小负 |
| RGB top25 | -0.011869 | +0.5414 | -0.011808 | +0.5303 | RGB 高误差区域明显改善 |
| RGB-only top25 | -0.011647 | +0.5477 | -0.011686 | +0.5459 | 即使没有 DINO 重叠，RGB-only 区域也改善 |
| DINO/RGB intersection top25 | -0.012438 | +0.5276 | -0.012121 | +0.4962 | 交集区域改善，但幅度与 RGB-only 接近 |
| DINO top25 | -0.000057 | -0.0188 | +0.000148 | -0.0572 | 纯看 DINO top-k 几乎没有收益 |
| DINO-only top25 | +0.004755 | -2.6560 | +0.004917 | -2.7183 | DINO 独有区域退化明显 |
| RGB broad top50 | -0.006183 | +0.3165 | -0.006164 | +0.3071 | broad RGB 候选内整体改善 |

判断：

- DINO rerank 的局部改善主要出现在 RGB 高误差区域，而不是 DINO-only 区域。DINO/RGB 交集区域改善存在，但与 RGB-only 区域改善幅度接近。
- DINO-only top-25% 区域 L1 反而升高，说明 DINO descriptor residual 目前不适合单独作为“应该增长”的空间 selector。
- 这支持继续做 `final top-m` 容量锁定实验：让 RGB 决定候选和容量，DINO 只改变候选内部排序；final-topm 判别结果见下一节。

## 2026-05-12 Phase 2：Final Top-M 容量锁定判别

新增 `vfm_rgb_rerank_final_topm` 配置开关和 `configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml`。该模式保留 `rgb_rerank` 的 broad RGB gate 和 DINO rerank score，但在实际 densification 前用 RGB broad reference score 计算同一步的参考候选数 `m`，再只从 broad candidate 内按 DINO-rerank 后的 score 取 top-m。这样每个 densification step 的增长容量由 RGB broad control 决定，DINO 只改变候选内部排序。

620-step smoke 已完成：

| Run | active_from | lambda | PSNR | SSIM | LPIPS | Gaussians | 输出 |
|---|---:|---:|---:|---:|---:|---:|---|
| final-topm l0.25 620 | 600 | 0.25 | 19.4840 | 0.4052 | 0.6290 | 63,443 | `output/0003/dino_rgb_rerank_finaltopm_l025_bicycle_620_r_auto` |
| final-topm l0.10 620 | 600 | 0.10 | 19.4752 | 0.4052 | 0.6280 | 63,446 | `output/0003/dino_rgb_rerank_finaltopm_l010_bicycle_620_r_auto` |

判断：

- 两组 train/render/metrics 均完成，说明 final-topm 的 scorer -> GaussianModel 引用分数传递链路健康。
- 620-step 点数与 RGB broad 620 的 63,439、非锁定 rerank 620 的 63,442 非常接近，符合“容量锁定只改变排序”的预期。
- 30k 正式判别已用双卡完成：GPU0 跑 `finaltopm l0.25 start7000`，GPU1 跑 `finaltopm l0.10 start9000`。

30k 全图结果：

| Run | lambda | active_from | PSNR | SSIM | LPIPS | Gaussians | vs RGB broad | 输出 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| RGB broad control | - | 0 | 25.3627 | 0.7656 | 0.2273 | 1,883,915 | - | `output/0003/rgb_broad_bicycle_30k_r_auto` |
| rerank l0.25 start7000 | 0.25 | 7000 | 25.3695 | 0.7663 | 0.2255 | 1,924,629 | +0.0068 / +0.0007 / -0.0018，+40,714 点 | `output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto` |
| rerank l0.10 start9000 | 0.10 | 9000 | 25.3556 | 0.7660 | 0.2261 | 1,907,193 | -0.0070 / +0.0004 / -0.0013，+23,278 点 | `output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto` |
| final-topm l0.25 start7000 | 0.25 | 7000 | 25.3564 | 0.7659 | 0.2266 | 1,896,839 | -0.0063 / +0.0003 / -0.0008，+12,924 点 | `output/0003/dino_rgb_rerank_finaltopm_l025_start7000_bicycle_30k_r_auto` |
| final-topm l0.10 start9000 | 0.10 | 9000 | 25.3692 | 0.7657 | 0.2270 | 1,893,008 | +0.0065 / +0.0001 / -0.0003，+9,093 点 | `output/0003/dino_rgb_rerank_finaltopm_l010_start9000_bicycle_30k_r_auto` |

final-topm 局部区域诊断输出：

- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_finaltopm/summary.json`
- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_finaltopm/per_view.csv`

相对 RGB broad control 的 top-25 局部变化：

| Region | final-topm l0.25 ΔL1 | final-topm l0.25 ΔPSNR | final-topm l0.10 ΔL1 | final-topm l0.10 ΔPSNR | 解释 |
|---|---:|---:|---:|---:|---|
| all | +0.000065 | -0.0165 | +0.000006 | -0.0025 | 全图空间 L1/MSE 无正向 |
| RGB top25 | -0.011801 | +0.5515 | -0.011939 | +0.5666 | RGB 高误差区仍改善 |
| RGB-only top25 | -0.011662 | +0.5640 | -0.011821 | +0.5775 | RGB-only 改善幅度不低于交集 |
| DINO/RGB intersection top25 | -0.012159 | +0.5240 | -0.012243 | +0.5428 | 交集改善存在，但不构成 DINO 独立证据 |
| DINO top25 | +0.000204 | -0.0392 | +0.000193 | -0.0176 | 纯 DINO top-k 无收益 |
| DINO-only top25 | +0.005009 | -2.7328 | +0.005027 | -2.7022 | DINO 独有区域继续明显退化 |
| RGB broad top50 | -0.006082 | +0.3228 | -0.006219 | +0.3401 | broad RGB 候选内整体改善 |

判断：

- final-topm 实现达到了容量锁定目标：相对 RGB broad 只多 9k~13k 点，明显低于非锁定 rerank 的 +23k~41k 点。
- 容量锁定后没有出现稳定全图质量优势。l0.10 final-topm PSNR 小正，但 SSIM/LPIPS 基本贴近 RGB broad；l0.25 final-topm 则 PSNR 负向。
- 局部诊断仍显示 DINO-only top-25 区域退化，且 RGB-only 与 DINO/RGB intersection 的改善幅度接近。这说明当前 DINO descriptor residual 没有成为有效的 densification selector。
- 0003 DINO rerank 训练分支建议收束：保留诊断脚本、final-topm 实现和负结果证据；后续若继续 DINO，应转向新的监督目标或后验分析，而不是继续扫 lambda/start_iter/broad top-k。

## 2026-05-13 Phase 3：DINO Prune-Protect Only 设计

本轮不再让 DINO 参与增长，也不做 `DINO says redundant -> prune`。设计目标是验证一个更保守的问题：在 FastGS/RGB 已经认为某些 GS 多视角不一致、可能进入 final pruning 时，DINO descriptor residual 是否能保护一小部分语义/结构上仍有价值的 GS，减少误删。

新增配置：

- `configs/experiments/0003_dino_descriptor_prune_protect_only.yaml`

关键参数：

```text
vfm_importance_mode = rgb_only
vfm_weight = 0.0
vfm_active_from_iter = 15001
vfm_prune_protect_weight = 0.25
vfm_prune_protect_mode = rgb_prune_candidate
vfm_prune_protect_rgb_min_score = 0.90
vfm_prune_protect_min_count = 5
vfm_prune_protect_power = 2.0
```

含义：

- 15k 之前的 densification 完全由 FastGS/RGB 决定；DINO 不参与增长。
- 15k 之后才激活 DINO，因此只影响 18k/21k/24k/27k 的 final pruning score。
- `rgb_prune_candidate` 只在 `rgb_pruning >= 0.90` 的候选中允许 DINO 保护，避免 DINO 对全局 pruning score 产生无锚点扰动。
- 第一轮对照 high-res `bicycle` 的 5090 FastGS big baseline：25.2569 / 0.7553 / 0.2450，1,560,209 个 Gaussians，输出位于 `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_single_gpu0/bicycle/fastgs_big_densify100_30k_r_auto`。

待跑顺序：

1. 620-step preflight smoke：只验证配置和 cache preflight，不触发 DINO protection。
2. 18.1k prune-path smoke：触发 iteration 18000 的 final pruning，确认 `rgb_prune_candidate` protection 分支真实运行。
3. 30k pilot：与 high-res FastGS big bicycle baseline 对比 PSNR/SSIM/LPIPS、Gaussian count 和日志中的 pruning 行为。

成功标准：相对 FastGS big baseline 至少 SSIM/LPIPS 不退，PSNR 不明显下降，Gaussian count 不显著增加；若只增加点数或保护导致剪不动但质量不升，则 prune-protect 分支收束为负结果。
