# 0003 复盘

## 当前判断

0001 最大的问题不是“DINO 完全无效”，而是证据链不够闭合：它证明了 DINO 介入 densification 后训练结果能变好，却没有充分证明 DINO descriptor residual map 与当前 FastGS 的重建瓶颈对齐。0002 里的 token-edge overlap 只能提示这个风险，不能直接解释 0001 descriptor 结果。

2026-05-12 的 Phase 0 已补上训练时同款 `render-vs-GT DINO cosine error map` 诊断。high-res `bicycle` 上，DINO descriptor residual 的 top-k 与 RGB 高误差 top-k 只有弱重叠：top-25% IoU 为 0.155~0.164，随机基线为 0.143；top-10% IoU 为 0.064~0.071，随机基线为 0.053。`w1600` 并没有优于 `w224/w518`，说明单纯提高 token 粒度不是解法。

Phase 1 第一轮已实现 `rgb_broad` 和 `rgb_rerank` 两个 importance mode，并完成 high-res `bicycle` 620-step smoke。RGB broad control 为 19.4699 / 0.4046 / 0.6282、63,439 点；DINO rerank l0.25 为 19.4483 / 0.4051 / 0.6281、63,442 点。该结果只证明链路健康，不证明质量收益；真正判断要进入 30k matched pilot。

Phase 2 首轮 high-res `bicycle` 30k matched pilot 已完成。RGB broad control 为 25.3627 / 0.7656 / 0.2273、1,883,915 点；DINO rerank l0.25 start9000 为 25.3538 / 0.7659 / 0.2260、1,915,967 点。相对 matched control，DINO rerank 是 -0.0088 PSNR、+0.0003 SSIM、LPIPS -0.0013、+32,052 点。它证明链路稳定，但还不是明确正向。

这会带来两种混淆：

- 如果 DINO top-k 区域和 RGB 高误差区域不重叠，质量提升可能来自训练轨迹、容量变化或间接正则，而不是精准结构引导。
- 如果 DINO token 太粗，再上采样成像素级 map，Gaussian 复制会落在大块区域内，无法证明引导在结构边界附近精确生效。

## 0001 Token 粒度问题

0001 descriptor 主线实际使用的 token grid 是 `10x16` 量级。它在 `-r 8` 图像上已经偏粗，在 high-res 1.6K 复验中则明显过粗。`vfm_descriptor_token_smooth_kernel=3` 会在 `10x16` grid 上进一步平滑，相当于把少数 patch 的响应扩成大块区域。

这可能解释 top-k25 的一部分不稳定性：它可能在“粗结构区域”增加了 Gaussian，而不是在真正的当前误差瓶颈处做精细复制。但这仍是假设，必须用真实 descriptor residual overlap 验证，不能由 token-edge 低重叠直接推出。

Phase 0 的结果进一步收紧了这个判断：`w224` 的 IoU 略高于 `w518/w1600`，说明 0001 的问题不只是 token 太粗；DINO descriptor residual 本身与 photometric error 的目标就不一致。后续不应继续把“更密 DINO tokens”作为主要修复方向。

## DINO 特性重新理解

DINOv2 patch token 的优势是语义/结构一致性和一定程度的光照、纹理不变性；弱点是它不天然等价于 RGB 重建误差。对 Gaussian Splatting 来说，训练主损失仍是 photometric/SSIM，DINO 如果不被当前误差锚定，就可能去强调“语义上重要但已经重建得还可以”的区域。

因此 DINO 更适合作为二级决策，而不是一级候选生成器。FastGS/RGB 先回答“哪里当前重建不好”，DINO 再在这些区域里回答“哪里更有语义结构价值”。

因此 0003 的 DINO 用法应从：

```text
DINO says important -> densify
```

改为：

```text
RGB/SSIM broad candidate -> DINO rerank/protect -> densify
```

## 优先修改方向

1. 训练时同款 DINO residual map 已落地
   `scripts/diagnose_dino_descriptor_residual.py` 已能读取 baseline render、GT/cache tokens，复现 `dinov2_descriptor_cosine` 的 patch error，并输出 per-view CSV 与 summary JSON。`w224/w518/w1600` 三尺度已经在 `bicycle` 上完成 top-25/top-10 overlap 诊断。

2. 不再把高分辨率 patch tokens 当作主修复
   `w1600` 的 overlap 和 Spearman 都没有改善。高分辨率 tokens 仍可作为后续局部结构指标或可视化工具，但不是 0003 第一训练候选的核心变量。

3. RGB 放宽候选 + DINO rerank
   第一版训练候选已落地为 `rgb_broad` 和 `rgb_rerank`。当前实现用 `loss_thresh=0.05` 放宽 RGB/FastGS 计数，再用 `vfm_rgb_broad_topk=0.50` 限定 broad candidate；`rgb_rerank` 在候选内采用 `RGB_importance * (1 + lambda * normalize(DINO residual))`。Phase 0 里 DINO top-k 有 52%~55% 落在 RGB top-50% broad candidate 内，这为 rerank 提供了比裸 top-k 更合理的入口。下一步要通过 30k matched control 验证它是否真带来质量或容量收益。

4. 后期介入
   DINO 不应从早期结构尚未成型时介入。第一组训练扫描建议 `DINO_start_iter = 7000/9000/11000`，仍保持 `densify_until_iter = 15000`，只在 densification 后半段做二次筛选。

5. Patch-aware map
   初始阶段至少改成 nearest upsample 或 token-cell mask，避免双线性插值制造虚假的亚 token 精度。后续可改为 token-grid 级 Gaussian visibility 聚合。

6. DINO prune 先做保护，不做主动删除
   若二次筛选有效，再考虑 DINO 辅助 pruning。第一版只做 semantic protection：`RGB pruning says bad AND DINO says important -> protect`。主动用 DINO 删除多余 GS 风险更高，暂不作为 0003 第一候选。

7. 局部指标必须入表
   每轮除了 PSNR/SSIM/LPIPS，还要报告 DINO top-k 区域、RGB 高误差区域和二者交集区域的 L1/LPIPS 改善。

## 对 0001 的重新定位

0001 可以保留为“DINO 介入 densification 有正向潜力”的工程证据，但不应继续作为“DINO metric map 已经有效定位结构瓶颈”的证据。0003 需要重建这个中间环节。

## 下一步

继续完成 start_iter 扫描，但保持保守解释。下一轮用双卡并行跑 `DINO_start_iter=7000/11000`，仍对照同一个 RGB broad 30k control；若两者仍是 PSNR 负、点数增、SSIM/LPIPS 仅微正，则暂停扩场景，改为 `lambda=0.10` 或显式 final top-m/局部指标诊断。
