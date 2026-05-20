# 0004 Late Scene-Adaptive Auxiliary 复盘

## 结论

两轮 0004 结果说明：Depth Anything `rgb_prune_auto_topk` 作为 late prune-protect 辅助器，确实比 0002 早期 densify/rerank 更接近中性，但还不能作为默认策略。`start24000` 把 `room` 的 30k 负向从 -0.2219 PSNR 收窄到 -0.0498，并把 24k -> 30k late gain 差值从 -0.1589 修复到 -0.0004；但 6 场景平均仍为 -0.0340 PSNR、-0.00029 SSIM、LPIPS +0.00001，且 `bonsai/kitchen` 是新的主要负例。

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

## 第二轮复盘

`start24000` 只在 24k/27k pruning 中启用 protect，验证了 timing 问题是真实存在的：

- `room` 明显修复：30k 负向从 -0.2219 收窄到 -0.0498 PSNR；24k -> 30k late gain 差值几乎归零。
- `stump/counter` 继续是小正向：分别 +0.0357 和 +0.0375 PSNR。
- `bicycle` 近中性：-0.0103 PSNR，LPIPS 略好。
- `bonsai/kitchen` 变成主要负例：分别 -0.1283 和 -0.0887 PSNR。

因此 `start24000` 不是默认解，而是把问题从“过早介入伤害 room”推进到“prior/protect 排序在部分室内场景仍不可靠”。这说明 late-only 是必要条件，不是充分条件。

## 风险点

- 辅助信号仍然可能和 RGB 误差错位。
- scene-adaptive 可能只是更复杂的超参表。
- 预算约束一旦太硬，质量可能直接掉回 baseline 以下。

## 下一步

下一步如果继续，应只做低成本 final-only 小扫，不再启用 checkpoint curve。优先级：

- 更低 protect weight，例如 0.10 或 0.15。
- 更窄 auto-topk 上限，例如 max 0.005。
- 只跑 `room/bonsai/counter/kitchen` 这四个最有判别力的场景，训练完只评测 30k。
- 如果 `bonsai/kitchen` 仍明显负向，就停止 Depth Anything prune-protect 在线方向。

评测方式也要收敛：start24000 这轮保留 checkpoint curve 是为了验证介入时机；后续除非专门做曲线诊断，否则不再每 2k render/metric，统一回到 final-only 评测。
