# 0003 DINO Guidance Reboot

## 核心假设

0001 证明了 DINO descriptor guidance 可以通过 densification 改变 FastGS 训练结果，但它没有充分证明 DINO metric map 精准命中了当前渲染的 RGB/LPIPS 瓶颈。0003 重新审视 DINO 的 token 粒度、语义不变性、介入时机和 metric-map 构造方式，目标是先导出训练时同款 `render-vs-GT DINO cosine error map`，再把 DINO 从“单独决定哪里增长”的一级 prior 改成“在 RGB 放宽候选内部二次筛选/排序”的后期辅助信号。

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

这不能直接否定 0001 的 descriptor residual，因为 token-edge 不是 descriptor residual；同时该诊断脚本如果直接读取 3D descriptor cache，只能用 channel norm 做粗略 proxy，也不等价于训练时的 render-vs-GT cosine error。更准确的说法是：token-edge 结果只证明裸结构显著性 prior 不能自动当成 RGB error proxy，并提示 0003 必须直接诊断真实 descriptor residual，而不是用 token-edge 低重叠来解释 0001 的 descriptor 训练结果。

0003 的第一原则是：先诊断训练时同款 DINO descriptor residual map，再训练。不要继续只扫裸 DINO top-k、importance weight、candidate cap 或 staged target。若 DINO 与 RGB 全局 top-k 不重合，第一训练候选也不应让 DINO 单独拉入新区域，而应让 RGB 先给出宽候选，再由 DINO 在候选内部 rerank。

## 新引导方向

1. 训练时同款 residual 诊断
   对已有 baseline render 重新计算 `rendered DINO tokens vs GT DINO tokens` 的 cosine residual，导出与 `dinov2_descriptor_cosine` 训练后端一致的 DINO metric map，再与 RGB error、SSIM/LPIPS proxy、depth prior 和 GT edge 做 overlap/correlation。已有 `output/0001/vfm_cache/*_dinov2_vits14` 保存 GT/source image 的 DINO patch tokens；诊断只需要对 render 图重新跑 DINO，若本机缺 checkpoint 则重新下载一次 DINOv2 权重。

2. 提高 token 粒度
   在 high-res 场景上比较 `max_width=224/518/1600` 三种 patch-token 粒度，先用少数场景评估存储和速度。对 1.6K 图像，`max_width=1600` token grid 约 `75x114`，比 0001 的 `10x16` 更适合做局部引导。若后续实现多层特征读取，再比较 early/mid/late descriptor；早层可能更贴近局部纹理，晚层更偏语义稳定性。

3. RGB 放宽候选 + DINO 二次筛选
   第一版训练候选不再是裸 `topk(DINO)`，也不优先使用严格 `topk(RGB) AND topk(DINO)`。更稳的候选是让 RGB/FastGS 先给出较宽增长候选，再让 DINO 在这些候选内部排序：

   ```text
   candidate = topk(RGB error, broad = 0.40~0.50)
   score = RGB_importance * (1 + lambda * normalize(DINO residual))
   final = topm(candidate, by = score)
   ```

   或更保守的二阶段形式：

   ```text
   candidate = topk(RGB error, broad = 0.40~0.50)
   final = topm(candidate, by = normalize(DINO residual))
   ```

   这样 DINO 只在当前重建确实困难的区域内改变优先级，不允许单独把 RGB 低误差区域拉进 densification。

   2026-05-12 已落地第一版实现：`rgb_broad` 作为 RGB broad matched control，`rgb_rerank` 在 broad candidate 内用 DINO descriptor residual 调整 importance。当前实现暂不做独立 `final_topm` 截断，而是沿用 FastGS 的 `densify_metric_thresh` 选择；620-step smoke 已证明链路健康，30k pilot 后再决定是否需要显式 top-m。

4. 后期介入
   DINO 不应从早期结构尚未成型时介入。第一组训练扫描使用 `DINO_start_iter = 7000/9000/11000`，保持 `densify_until_iter = 15000`。前半段让 FastGS/RGB 建立几何和外观 scaffold，后半段再让 DINO 在 RGB 候选中做语义结构二次筛选。

5. Patch-aware Gaussian 计数
   避免把低分辨率 patch map 双线性上采样成看似精细的像素 map。初始实现可以先用 nearest upsample 保持 token cell 边界；后续再考虑把 Gaussian visibility 直接聚合到 patch grid。

6. DINO 辅助裁剪只做后续保护型实验
   如果 densification rerank 成立，再研究 DINO pruning。第一版不让 DINO 主动删除 GS，而是做 semantic protection：`RGB pruning says bad AND DINO says important -> protect`。主动裁剪风险更高，只有在保护型实验明确正向后再考虑。

7. 局部指标优先
   DINO guidance 的成功标准不能只看全图 PSNR。每轮必须报告 DINO/RGB overlap、prior 区域 L1/LPIPS 改善、非 prior 区域变化和 Gaussian 增量分布。

## 阶段计划

| 阶段 | 目标 | 产物 | 是否训练 |
|---|---|---|---|
| Phase 0 | 导出训练时同款 render-vs-GT DINO cosine error map，并诊断 overlap / token 粒度 | diagnostic CSV/JSON/overlay；已完成 bicycle w224/w518/w1600 | 否 |
| Phase 1 | 实现 RGB-broad candidate + DINO rerank + late activation，620-step 验证链路 | config + smoke logs；已完成 bicycle 620 | 是，短跑 |
| Phase 2 | high-res `bicycle/stump/treehill/bonsai` 30k pilot，扫描 start_iter/lambda/broad top-k | bicycle start_iter 和 lambda=0.10 已完成；下一步局部诊断或 final top-m | 是 |
| Phase 3 | 若 Phase 2 成立，再做 DINO prune-protect，不做主动 DINO pruning | config + pilot metrics | 是 |
| Phase 4 | 只在 pilot 成立后扩全数据集 | summary + policy | 是 |

## 初始成功标准

- 训练时同款 DINO residual 的全局 top-k overlap、以及在 RGB broad candidate 内的 DINO rerank 分布，都要明显好于随机 baseline；否则 DINO 不应继续作为 densification 主信号。
- RGB-broad + DINO rerank + late activation 在至少两个 high-res pilot 场景相对 matched baseline 三项质量不退，且 Gaussian 增量受控；相对 RGB-only broad candidate 需要质量不降或点数更省。
- 如果 DINO 指标只改善局部结构但不改善全图指标，应改写目标为局部结构质量或 selector，而不是继续包装成全图 PSNR 主线。

## 下一步

Phase 2 high-res `bicycle` 30k 已完成 `DINO rerank lambda=0.25, start_iter=7000/9000/11000` 和 `lambda=0.10, start_iter=7000/9000` 扫描。`lambda=0.25 start7000` 相对 RGB broad control 为 +0.0068 PSNR、+0.0007 SSIM、LPIPS -0.0018，但多 40,714 点；`lambda=0.10` 只省 8k~11k 点，PSNR 仍低于 RGB broad control。下一步暂停扩多场景，转向显式 final top-m 或局部指标诊断，先证明 DINO rerank 是否真的改善 DINO/RGB 交集区域。
