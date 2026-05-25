# 0017 Results

## Status

Completed. Started in tmux at `2026-05-25 11:03:09 +0800`; all sessions exited and merge completed.

Sessions:

- `0017_desc_clone16k_g0`
- `0017_desc_clone16k_g1`
- `0017_desc_clone16k_merge`

0017 追踪 0016 补充 smoke 中唯一有正信号的 `desc_clone_16k`：

- MipNeRF360 full9。
- 从 FastGS baseline checkpoint-curve 的 16K PLY 接续。
- DINO descriptor 只控制 clone，不控制 split。
- 关闭 densify prune、final prune、opacity reset。
- 续跑 5K 到 21K。
- 保存并评测 16K / 17K / 18K / 19K / 20K / 21K。
- 训练完成后统一 render + metric。

## Outputs

```text
output/0017/descriptor_clone16k_full9/mipnerf360_combined/summary.csv
output/0017/descriptor_clone16k_full9/mipnerf360_combined/comparison_start_to_eval.csv
output/0017/descriptor_clone16k_full9/mipnerf360_combined/aggregate_by_iteration.csv
```

完整性：

- `summary.csv`：54 行，即 9 scenes x 6 eval points。
- `comparison_start_to_eval.csv`：45 行，即 9 scenes x 5 eval deltas。
- `aggregate_by_iteration.csv`：5 行，即 +1K / +2K / +3K / +4K / +5K。

## Aggregate Vs 16K Start

相对 16K 起点，descriptor clone-only 是稳定正向的：

| eval | scenes | avg dPSNR | avg dSSIM | avg dLPIPS | avg dGS | PSNR+ | SSIM+ | LPIPS better |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1K | 9 | +0.0714 | +0.00059 | -0.00113 | +9,113 | 5/9 | 6/9 | 8/9 |
| +2K | 9 | +0.1043 | +0.00102 | -0.00199 | +12,917 | 8/9 | 9/9 | 9/9 |
| +3K | 9 | +0.1236 | +0.00131 | -0.00253 | +15,452 | 8/9 | 9/9 | 9/9 |
| +4K | 9 | +0.1367 | +0.00152 | -0.00286 | +17,210 | 7/9 | 9/9 | 9/9 |
| +5K | 9 | +0.1466 | +0.00159 | -0.00305 | +18,551 | 8/9 | 9/9 | 9/9 |

## Final Per-Scene

| scene | PSNR 16K -> 21K | dPSNR | dSSIM | dLPIPS | dGS |
|---|---:|---:|---:|---:|---:|
| bicycle | 25.0880 -> 25.1492 | +0.0612 | +0.00176 | -0.00386 | +12,271 |
| flowers | 21.5731 -> 21.5989 | +0.0258 | +0.00284 | -0.00412 | +47,973 |
| garden | 27.4569 -> 27.5138 | +0.0568 | +0.00097 | -0.00189 | +2,304 |
| stump | 27.1287 -> 27.1461 | +0.0174 | +0.00046 | -0.00216 | +22,120 |
| treehill | 22.8233 -> 22.8198 | -0.0035 | +0.00075 | -0.00357 | +81,551 |
| room | 31.5913 -> 31.9665 | +0.3751 | +0.00229 | -0.00347 | +562 |
| counter | 29.1052 -> 29.2768 | +0.1716 | +0.00162 | -0.00296 | +165 |
| kitchen | 31.8556 -> 32.1473 | +0.2917 | +0.00205 | -0.00270 | +3 |
| bonsai | 32.1146 -> 32.4380 | +0.3234 | +0.00154 | -0.00275 | +12 |

室内场景信号最干净：room / counter / kitchen / bonsai 都有明显 PSNR 增长，且几乎不增点。室外场景也大多正向，但 flowers / stump / treehill 的增点更高，其中 treehill 是唯一 PSNR 轻微负向的场景。

## Baseline Curve Check

只看相对 16K 起点会高估这条线。和正常 FastGS checkpoint curve 的同迭代结果对齐后，descriptor clone-only 仍然落后：

| comparison | avg dPSNR | avg dSSIM | avg dLPIPS | avg dGS | note |
|---|---:|---:|---:|---:|---|
| 0017 18K - baseline 18K | -0.0439 | -0.00103 | +0.00184 | +12,917 | 9/9 PSNR 不如 baseline |
| 0017 20K - baseline 20K | -0.0424 | -0.00131 | +0.00214 | +203,030 | 只有 flowers PSNR 基本打平，其他 8/9 低于 baseline |
| 0017 21K - baseline 20K | -0.0325 | -0.00124 | +0.00194 | +204,371 | 21K 仍低于 baseline 20K |
| 0017 21K - baseline 22K | -0.0893 | -0.00184 | +0.00301 | +215,526 | 与更接近的后续 baseline 差距扩大 |

这说明 descriptor clone-only 可以让 frozen 16K 起点继续变好，但不能替代正常 FastGS 在 16K-21K 的 RGB-driven continuation。

## Decision

0017 是一个正向排雷结果，但不是默认化结果。

结论：

- 保留 `descriptor clone@16K` 作为有效 late refinement 信号：它不会像 token-edge split-only 那样崩盘，且 full9 相对 16K 起点平均正向。
- 不把 `vfm_only + clone_only + prune_off` 当作 FastGS 16K 后默认 continuation：同步 baseline 仍更好，且 20K 时平均多约 20.3 万 Gaussian。
- 如果继续推进，下一步应保留 RGB/FastGS 原始候选或加容量 cap，让 descriptor 只做 clone 候选的 rerank/protect；重点盯 indoor 场景的低增点高收益，以及 treehill / flowers 的容量失控风险。
