# 0004 Late Scene-Adaptive Auxiliary 复盘

## 结论

尚无正式结果。

## 目前判断

0004 的价值不在于继续证明某个 prior 很强，而在于验证一个更窄的使用方式是否成立：晚期、场景自适应、预算约束、只做辅助。

`fastgs_big_baseline_checkpoint_curve` 的第一条启发是：late intervention 的时间不能固定成单一常数。MipNeRF360 里 `room/counter/kitchen/bonsai` 到 24k 后仍有可见 PSNR 增长；`flowers/treehill/garden/bicycle` 后期收益很薄；`stump` 的 PSNR 在 22k 达峰而 LPIPS 继续改善。0004 如果只在 30k 对比，可能会把“prior 是否有效”和“baseline 自然后期收益/回落”混在一起。

因此第一轮 0004 pilot 的判据应是曲线型而不是单点型：

- 20k/24k/30k 三个窗口都要报告。
- 判断 prior 时至少要和同场景 baseline curve 的自然增量比较。
- 若 prior 只带来小于约 0.02 PSNR 的变化，应默认视为平台期噪声，除非 LPIPS/GS 同时清晰改善。
- `stump` 这类 PSNR 和 LPIPS 分歧场景，要明确选择目标：若实验目标是视觉质量，LPIPS 改善可以记录；若目标是 PSNR/SSIM，不应被后期 LPIPS 小收益误导。

## 风险点

- 辅助信号仍然可能和 RGB 误差错位。
- scene-adaptive 可能只是更复杂的超参表。
- 预算约束一旦太硬，质量可能直接掉回 baseline 以下。

## 下一步

先把小 pilot 跑完，再决定要不要扩到全量场景。
