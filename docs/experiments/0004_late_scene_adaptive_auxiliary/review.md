# 0004 Late Scene-Adaptive Auxiliary 复盘

## 结论

第一轮 0004 结果说明：Depth Anything `rgb_prune_auto_topk` 作为 late prune-protect 辅助器，确实比 0002 早期 densify/rerank 更接近中性，但还不能作为默认策略。6 场景 30k 平均相对 baseline 为 -0.0242 PSNR、-0.00014 SSIM、LPIPS +0.00014、平均多 529 个 Gaussian；其中 `room` 的 -0.2219 PSNR 是明确负例。

## 目前判断

0004 的价值不在于继续证明某个 prior 很强，而在于验证一个更窄的使用方式是否成立：晚期、场景自适应、预算约束、只做辅助。

`fastgs_big_baseline_checkpoint_curve` 的第一条启发是：late intervention 的时间不能固定成单一常数。MipNeRF360 里 `room/counter/kitchen/bonsai` 到 24k 后仍有可见 PSNR 增长；`flowers/treehill/garden/bicycle` 后期收益很薄；`stump` 的 PSNR 在 22k 达峰而 LPIPS 继续改善。0004 如果只在 30k 对比，可能会把“prior 是否有效”和“baseline 自然后期收益/回落”混在一起。

因此第一轮 0004 pilot 的判据应是曲线型而不是单点型：

- 20k/24k/30k 三个窗口都要报告。
- 判断 prior 时至少要和同场景 baseline curve 的自然增量比较。
- 若 prior 只带来小于约 0.02 PSNR 的变化，应默认视为平台期噪声，除非 LPIPS/GS 同时清晰改善。
- `stump` 这类 PSNR 和 LPIPS 分歧场景，要明确选择目标：若实验目标是视觉质量，LPIPS 改善可以记录；若目标是 PSNR/SSIM，不应被后期 LPIPS 小收益误导。

## 第一轮复盘

`start15001` 实际在 18k/21k/24k/27k 四次 pruning 中启用 protect。它的表现是典型的“减伤器还没变成增益器”：

- `bicycle` 基本中性，30k +0.0067 PSNR，LPIPS 小幅改善。
- `stump` 30k +0.0252 PSNR，但 SSIM/LPIPS 变差；它更像是减轻 PSNR 后期回落，而不是全面提质。
- `counter` 30k +0.0452 PSNR，但 SSIM/LPIPS 没有同步改善。
- `bonsai` 30k +0.0703 PSNR，但 16k/20k/24k -> 30k 的 late gain 全部低于 baseline，说明单点正向主要来自早期曲线偏移，不能直接归因给 protect。
- `kitchen` 24k 后窗口略正，但最终仍 -0.0706 PSNR。
- `room` 30k -0.2219 PSNR，且 24k -> 30k 窗口比 baseline 少 +0.1589 PSNR，是当前最重要的失败样本。

这轮结果支持一个更窄的假设：辅助器如果要存在，应该更晚、更少次地介入。`room` 在 baseline 里 24k -> 30k 仍有 +0.1058 PSNR，18k/21k 的保护可能过早固定了本该继续被 pruning/重分配的 Gaussian 生命周期。

## 风险点

- 辅助信号仍然可能和 RGB 误差错位。
- scene-adaptive 可能只是更复杂的超参表。
- 预算约束一旦太硬，质量可能直接掉回 baseline 以下。

## 下一步

下一轮只改 `vfm_active_from_iter`：从 15001 改到 24000，跳过 18k/21k，只在 24k/27k 生效。判断标准：

- 若 `room` 的负向明显收窄，同时 `stump/counter/bicycle` 不丢掉已有小收益，则继续做更轻权重或更窄 top-k 的 late-only sweep。
- 若 `room` 仍大幅负向，说明 Depth Anything prune-protect 的排序本身仍和 FastGS pruning 目标错位，应停止把它作为在线辅助器。
- 若平均只在 +/-0.02 PSNR 内波动且 LPIPS/GS 没有同步收益，则默认视为平台期噪声。

评测方式也要收敛：start24000 这轮保留 checkpoint curve 是为了验证介入时机；后续除非专门做曲线诊断，否则不再每 2k render/metric，统一回到 final-only 评测。
