# 0003 复盘

## 当前判断

0001 最大的问题不是“DINO 完全无效”，而是证据链不够闭合：它证明了 DINO 介入 densification 后训练结果能变好，却没有充分证明 DINO descriptor residual map 与当前 FastGS 的重建瓶颈对齐。0002 里的 token-edge overlap 只能提示这个风险，不能直接解释 0001 descriptor 结果。

2026-05-12 的 Phase 0 已补上训练时同款 `render-vs-GT DINO cosine error map` 诊断。high-res `bicycle` 上，DINO descriptor residual 的 top-k 与 RGB 高误差 top-k 只有弱重叠：top-25% IoU 为 0.155~0.164，随机基线为 0.143；top-10% IoU 为 0.064~0.071，随机基线为 0.053。`w1600` 并没有优于 `w224/w518`，说明单纯提高 token 粒度不是解法。

Phase 1 第一轮已实现 `rgb_broad` 和 `rgb_rerank` 两个 importance mode，并完成 high-res `bicycle` 620-step smoke。RGB broad control 为 19.4699 / 0.4046 / 0.6282、63,439 点；DINO rerank l0.25 为 19.4483 / 0.4051 / 0.6281、63,442 点。该结果只证明链路健康，不证明质量收益；真正判断要进入 30k matched pilot。

Phase 2 high-res `bicycle` 30k matched pilot 已完成 `lambda=0.25` start_iter 扫描和 `lambda=0.10` 复核。RGB broad control 为 25.3627 / 0.7656 / 0.2273、1,883,915 点；`lambda=0.25 start7000` 为 25.3695 / 0.7663 / 0.2255、1,924,629 点，是唯一三项质量小幅正向的 DINO rerank 点，但只多 +0.0068 PSNR 且多 40,714 点。`lambda=0.10 start7000/start9000` 分别为 25.3519 / 0.7660 / 0.2262、1,913,988 点和 25.3556 / 0.7660 / 0.2261、1,907,193 点；低 lambda 省点有限且 PSNR 仍低于 RGB broad control。它证明链路稳定，但还不是可扩场景的明确正向。

随后完成的局部区域诊断把这一点进一步收紧：以 RGB broad control 构造固定 mask 后，DINO/RGB top-25 IoU 为 0.1627、top-10 IoU 为 0.0716。相对 RGB broad control，`lambda=0.25 start7000` 在 RGB top-25 区域 L1 改善 -0.011869、PSNR +0.5414，在 DINO/RGB intersection top-25 区域 L1 改善 -0.012438、PSNR +0.5276；但在 DINO-only top-25 区域 L1 反而 +0.004755、PSNR -2.6560。`lambda=0.10 start9000` 也呈现同样方向。这说明当前 DINO rerank 的局部收益主要来自 RGB high-error 候选轨迹，DINO-only selector 并没有提供独立正证据。

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
   第一版训练候选已落地为 `rgb_broad` 和 `rgb_rerank`。当前实现用 `loss_thresh=0.05` 放宽 RGB/FastGS 计数，再用 `vfm_rgb_broad_topk=0.50` 限定 broad candidate；`rgb_rerank` 在候选内采用 `RGB_importance * (1 + lambda * normalize(DINO residual))`。Phase 0 里 DINO top-k 有 52%~55% 落在 RGB top-50% broad candidate 内，这为 rerank 提供了比裸 top-k 更合理的入口。30k start_iter/lambda 扫描显示它最多产生弱质量收益，且收益薄、容量更高。

4. 后期介入
   DINO 不应从早期结构尚未成型时介入。第一组训练扫描已覆盖 `DINO_start_iter = 7000/9000/11000`，仍保持 `densify_until_iter = 15000`，只在 densification 后半段做二次筛选。当前 `lambda=0.25 start7000` 最好，说明过晚介入未带来更强收益；`lambda=0.10` 没有修复容量/质量权衡。后续不应继续扫相邻 start_iter/lambda。

5. Patch-aware map
   初始阶段至少改成 nearest upsample 或 token-cell mask，避免双线性插值制造虚假的亚 token 精度。后续可改为 token-grid 级 Gaussian visibility 聚合。

6. DINO prune 先做保护，不做主动删除
   DINO rerank 分支已经显示 DINO-only 区域会退化，因此 pruning 方向只保留保守保护实验，不把 DINO 当主动删除信号。Phase 3 使用 `RGB pruning says bad AND DINO says important -> protect`，并限制在 RGB pruning high-score 候选中生效，等价于先由 FastGS/RGB 给出可删 proposal，再用 DINO 做 proposal 内保护。主动用 DINO 删除多余 GS 风险更高，暂不作为 0003 候选。

7. 局部指标必须入表
   已新增 `scripts/diagnose_0003_local_regions.py`，每轮除了 PSNR/SSIM/LPIPS，还要报告 DINO top-k 区域、RGB 高误差区域和二者交集区域的 L1/PSNR 改善。当前 LPIPS 工具只返回全图标量，暂不伪造空间 LPIPS map。

## 对 0001 的重新定位

0001 可以保留为“DINO 介入 densification 有正向潜力”的工程证据，但不应继续作为“DINO metric map 已经有效定位结构瓶颈”的证据。0003 需要重建这个中间环节。

## 下一步

暂停扩多场景，也暂停相邻 lambda/start_iter 扫描。局部指标已经显示 DINO-only 区域退化，因此最后保留一个关键判别实验：显式 final top-m，让 DINO rerank 只改变候选排序但不增加最终 densification 容量。该实现、620-step smoke 和 30k 双卡判别已完成。final-topm 确实把点数压回接近 RGB broad control，但全图只有弱混合信号，DINO-only top25 仍然退化。0003 的 DINO rerank 训练分支应暂时收束为“当前 DINO descriptor residual 不适合作为 FastGS RGB 瓶颈 selector”。

Phase 3 转向 DINO prune-protect only。新增 `configs/experiments/0003_dino_descriptor_prune_protect_only.yaml`，使用 `vfm_importance_mode=rgb_only`、`vfm_weight=0.0`、`vfm_active_from_iter=15001`，保证 DINO 不参与 15k 前 densification，只在后期 final pruning 中保护 `rgb_pruning >= 0.90` 的候选。这个实验回答的问题更窄：DINO 能否减少 RGB final pruning 的误删，而不是 DINO 能否发现应该删除或应该增长的位置。若该分支仍只带来点数增加、质量不升，DINOv2 descriptor 就不适合作为当前 FastGS per-Gaussian lifecycle signal。

Phase 3 prune-protect only 已完成 620-step preflight、18.1k prune-path smoke 和 high-res `bicycle` 30k pilot。30k 指标为 25.2519 / 0.7554 / 0.2449、1,555,224 点，相对 5090 FastGS big baseline 25.2569 / 0.7553 / 0.2450、1,560,209 点基本等同。四次 final-pruning 日志分别只有 `protected=2 / rgb_candidates=2`、`1/1`、`1/1`、`1/1`，说明 DINO protection 分支确实触发，但在 `rgb_pruning >= 0.90` 的安全 gate 下几乎没有作用空间。

这轮结果把 pruning 方向也收紧了：`RGB high-prune candidate -> DINO protect` 是安全的，但当前候选定义太窄，无法证明 DINOv2 descriptor 能改善 pruning 决策；若直接扩多场景，预计只会得到接近 baseline 的 no-op 结果。

随后完成的 threshold gate 复核进一步排除了一个简单修复：把 `vfm_prune_protect_rgb_min_score` 从 0.90 放宽到 0.80/0.70 后，18.1k prune-path smoke 仍然只有 `protected=1 / rgb_candidates=1`。这说明 RGB pruning score 在当前实现中并不是一个适合用绝对阈值切 proposal 的连续空间；至少在 `bicycle` 18k，threshold gate 放宽不会带来更多候选。

## Phase 3 结论

当前 prune-protect 不是负向崩坏，而是近 no-op：

- 它没有显著增加点数：比 baseline 少 4,985 个 Gaussians。
- 它没有显著改变质量：PSNR -0.0050，SSIM +0.0002，LPIPS -0.0000。
- 它没有足够候选：18k/21k/24k/27k 的 RGB high-prune candidate 总数只有 5 个。
- threshold 放宽无效：0.80/0.70 仍然各只有 1 个 18k candidate。

因此这条分支不能作为“DINOv2 可辅助 pruning”的正证据。更准确的判断是：在最保守的 RGB proposal gate 下，DINOv2 descriptor 的保护信号无法获得足够决策权。

后续若继续 pruning 方向，应该先做候选空间诊断，而不是直接跑全数据集：

- 放宽 RGB pruning proposal 时不要再扫绝对阈值；0.80/0.70 已经没有扩大候选。下一步应改成每个 pruning step 固定 top-k/top-p RGB pruning candidate。
- 保持 DINO 只做保护，不做主动删除。DINO-only densify 已显示错位，主动 prune 的风险更高。
- 增加 per-view/区域可视化：检查被保护 GS 投影到哪些图像区域，是否落在边界、遮挡、细结构或 DINO/RGB 交集区域。
- 若放宽后候选足够但指标仍贴 baseline，应把 DINOv2 descriptor 从 GS lifecycle signal 中移出，只保留为离线诊断或语义可视化工具。
