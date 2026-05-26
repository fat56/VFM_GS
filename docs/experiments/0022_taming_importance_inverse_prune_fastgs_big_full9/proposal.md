# 0022 Taming Importance Inverse-Prune FastGS Big Full9

## 核心问题

0021 只替换了 FastGS clone/split densification importance，densify-stage prune 和 final-prune 仍使用 FastGS pruning score。结果显示质量平均正向，但 Gaussian 数和训练时间显著增加。

0022 要验证：

> 若 densify-stage prune 也改成更接近 Taming-3DGS 的 inverse-importance pruning，能否保留 0021 的质量收益，同时回收一部分 Gaussian 膨胀？

## 具体对照

固定 FastGS big 30K 训练流程，沿用 0021 的 Taming-style per-Gaussian importance。

- Baseline：0002 `fastgs_big` full9 baseline。
- 0021：Taming-style importance 只用于 clone/split，prune 仍为 FastGS pruning score。
- 0022：Taming-style importance 用于 clone/split；densify-stage prune 在 FastGS low-opacity / oversized candidates 中按 inverse Taming-style importance 采样删除。

保持不变：

- MipNeRF360 原图 1.6K auto-resolution 口径。
- `fastgs_big` scene overrides。
- densification interval = 100。
- clone / split gradient and scale gates。
- 15K 后 FastGS final-prune tail。
- scorer 的 Taming-style importance coefficients。

变更：

```yaml
densify_prune_score_source: importance_inverse
```

## 实现口径

当前 FastGS densify-stage prune 先构造 prune candidates：

- opacity below `min_opacity`
- screen/world size too large when `max_screen_size` is active

0022 不改变这些 candidate gates，只改变从候选中采样删除点的 score source：

```text
0021 / FastGS-style: remove probability roughly follows 1 / (1 - pruning_score)
0022 / Taming-like:  remove probability roughly follows 1 / (eps + importance_score)
```

因此，高 Taming-style importance 仍更容易进入 clone/split，低 Taming-style importance 更容易在 prune candidates 中被删除。这更接近 Taming-3DGS 原版“高分增殖、低分裁剪”的语义。

重要限制：

- 0022 仍不是完整 Taming-3DGS 复现。
- final-prune tail 仍保持 FastGS pruning score，因为直接把 high-importance score 接到 `pruning_score > 0.9` 会反向删除重要点。
- 当前 FastGS rasterizer 仍不暴露 Taming-3DGS full weighted accumulators，因此 importance 仍是 0021 的 available-terms approximation。

## 数据与指标

数据集：MipNeRF360 全 9 场景。

指标：

- PSNR
- SSIM
- LPIPS
- GS_num
- train_time

对照 baseline：

```text
output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv
```

## 判定

若 0022 接近 0021 的 PSNR/SSIM/LPIPS，但 GS_num 明显下降，说明 0021 的容量膨胀可以通过 Taming-like inverse prune 缓解。

若 0022 质量明显下降且 GS_num 回落，说明 0021 的收益主要来自宽松容量，Taming-style prune 会删掉后续仍有价值的点。

若 0022 仍继续增点，说明当前 Taming-style importance 的容量问题来自 clone/split source 本身，而不是 densify-stage prune score source。
