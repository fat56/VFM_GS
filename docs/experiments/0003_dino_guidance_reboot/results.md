# 0003 实验结果

## 当前状态

0003 已完成 Phase 0 第一轮 `bicycle` 诊断。当前结论是：训练时同款 DINO descriptor residual 与 RGB 高误差区域只有弱重叠，单纯把 token grid 从 `10x16` 提高到 `75x114` 没有改善 overlap。下一步不应回到裸 DINO top-k，而应实现 RGB 放宽候选内部的 DINO rerank。

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
| TBD | Phase 1 | bicycle | RGB-broad candidate + DINO rerank + late activation 620-step smoke | TBD | TBD |
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
