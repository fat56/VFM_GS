# 0003 DINO Guidance Reboot

## 核心假设

0001 证明了 DINO descriptor guidance 可以通过 densification 改变 FastGS 训练结果，但它没有充分证明 DINO metric map 精准命中了当前渲染的 RGB/LPIPS 瓶颈。0003 重新审视 DINO 的 token 粒度、语义不变性和 metric-map 构造方式，目标是把 DINO 从“单独的结构 prior”改成“由当前重建误差锚定的结构引导”。

## 0001 的关键事实

0001 主线 descriptor cache 大多使用：

- DINOv2 ViT-S/14，patch size 为 14。
- `images_8` cache，`max_width=224`。
- `dinov2_patchtokens`，`npy_float16`，每个 token 384 维。
- `bicycle` cache 首图 shape 为 `10x16x384`，source image 为 `618x411`。
- `stump` cache 首图 shape 也是 `10x16x384`，source image 为 `622x413`。
- 训练时 descriptor residual 还用了 `vfm_descriptor_token_smooth_kernel=3`，再上采样到渲染分辨率后做 top-k25。

因此在 `-r 8` 口径下，一个 DINO token 大约覆盖 `39x41` 个渲染像素；如果 high-res 1.6K 训练仍复用这个 `10x16` cache，一个 token 会被放大到约 `100x106` 个渲染像素。这个粒度对“像素级 densification map”来说非常粗。

0001 也做过更密的 cache：

- `max_width=518` 的 ViT-S/14 descriptor cache，`bicycle` 首图为 `24x37x384`，但它没有成为主线。
- high-res ViT-L/14 token-edge cache，`max_width=1600`，`bicycle` 首图为 `75x114`，但它保存的是 `dinov2_token_edge` 2D map，不是在线 render-vs-GT descriptor residual。

## 问题判断

`scripts/diagnose_prior_overlap.py` 在 0002 中显示，结构 prior top-k 与 RGB 高误差 top-k 的重叠并不高。尤其 high-res `bicycle` 上，DINO ViT-L token-edge top-25% 与 RGB error top-25% 的 IoU 只有 0.149，top-10% 只有 0.068。

这不能直接否定 0001 的 descriptor residual，因为 token-edge 不是 descriptor residual；同时该诊断脚本如果直接读取 3D descriptor cache，只能用 channel norm 做粗略 proxy，也不等价于训练时的 render-vs-GT cosine error。但它足以说明一个风险：DINO 结构显著区域不必然是当前 FastGS 的 RGB/LPIPS 瓶颈区域。

0003 的第一原则是：先诊断真实 DINO descriptor residual map，再训练。不要继续只扫 top-k、importance weight、candidate cap 或 staged target。

## 新引导方向

1. 真实 residual 诊断  
   对已有 baseline render 重新计算 `rendered DINO tokens vs GT DINO tokens` 的 cosine residual，导出真实 DINO metric map，再与 RGB error、SSIM/LPIPS proxy、depth prior 和 GT edge 做 overlap/correlation。

2. 提高 token 粒度  
   在 high-res 场景上优先测试 ViT-S/14 `max_width=1600` patch-token cache，先用少数场景评估存储和速度。对 1.6K 图像，token grid 约 `75x114`，比 0001 的 `10x16` 更适合做局部引导。

3. RGB 锚定的 DINO map  
   0003 不再默认使用 `topk(DINO)`。首选候选是：

   ```text
   metric = normalize(DINO residual)^beta * normalize(RGB error)^alpha
   ```

   或更保守的：

   ```text
   metric = topk(DINO residual) AND topk(RGB error)
   ```

   这样 DINO 只在当前重建确实困难的区域放大 densification，而不是在所有语义/结构显著区域复制 Gaussian。

4. Patch-aware Gaussian 计数  
   避免把低分辨率 patch map 双线性上采样成看似精细的像素 map。初始实现可以先用 nearest upsample 保持 token cell 边界；后续再考虑把 Gaussian visibility 直接聚合到 patch grid。

5. 局部指标优先  
   DINO guidance 的成功标准不能只看全图 PSNR。每轮必须报告 DINO/RGB overlap、prior 区域 L1/LPIPS 改善、非 prior 区域变化和 Gaussian 增量分布。

## 阶段计划

| 阶段 | 目标 | 产物 | 是否训练 |
|---|---|---|---|
| Phase 0 | 重放真实 DINO descriptor residual 并诊断 overlap | diagnostic CSV/JSON/overlay | 否 |
| Phase 1 | 实现 RGB-gated DINO metric map，620-step 验证链路 | config + smoke logs | 是，短跑 |
| Phase 2 | high-res `bicycle/stump/treehill/bonsai` 30k pilot | matched metrics + overlap | 是 |
| Phase 3 | 只在 pilot 成立后扩全数据集 | summary + policy | 是 |

## 初始成功标准

- 真实 DINO residual top-k 与 RGB error top-k 的 overlap 明显高于 token-edge prior；否则 DINO 不应继续作为全图质量提升主 prior。
- RGB-gated DINO 在至少两个 high-res pilot 场景相对 matched baseline 三项质量不退，且 Gaussian 增量受控。
- 如果 DINO 指标只改善局部结构但不改善全图指标，应改写目标为局部结构质量或 selector，而不是继续包装成全图 PSNR 主线。

## 下一步

新增一个诊断脚本，导出真实 DINO descriptor residual map，并在 high-res `bicycle` 上比较 `224/518/1600` 三种 token 粒度与 RGB error 的重叠。
