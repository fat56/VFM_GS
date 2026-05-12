# 0003 实验结果

## 当前状态

0003 已初始化，当前尚未新增训练结果。第一步是诊断，不是直接跑 30k。

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
| TBD | Phase 0 | bicycle | 训练时同款 render-vs-GT DINO cosine error overlap | TBD | TBD |
| TBD | Phase 0 | bicycle | 224/518/1600 token 粒度比较 | TBD | TBD |
| TBD | Phase 1 | bicycle | RGB-broad candidate + DINO rerank + late activation 620-step smoke | TBD | TBD |
| TBD | Phase 2 | bicycle/stump/treehill/bonsai | start_iter/lambda/broad top-k 30k pilot | TBD | TBD |
| TBD | Phase 3 | TBD | DINO prune-protect pilot | TBD | 只在 rerank 成立后推进 |

## 决策

0003 不继承 0001 的“DINO descriptor 已经定位清楚”作为前提。0001 的全图质量正向仍是有价值证据，但 0003 会先导出训练时同款 DINO residual map，验证它是否覆盖当前重建误差瓶颈。若 DINO/RGB 全局 top-k 不重合，第一训练候选改为 RGB 放宽候选内部的 DINO rerank，而不是裸 DINO top-k 或 DINO 主动扩候选。
