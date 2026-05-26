# 0021 Results

## 结论

0021 full9 已完成。相对 `fastgs_big` baseline，`taming_importance_fastgs_prune_densify100` 平均为：

| scenes | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | GS_num | dGS_num | train_time | dtrain_time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 28.0042 | +0.0452 | 0.825438 | +0.005170 | 0.203844 | -0.011813 | 1,588,581 | +426,795 | 199.31s | +39.61s |

逐场景看，PSNR 为 7/9 正向，SSIM 为 9/9 正向，LPIPS 为 9/9 改善；但 Gaussian 数和训练时间也是 9/9 增加。这个结果不支持“FastGS big 的质量优势主要、且必须来自原始 multi-view VCD/VCP importance”这个强假设：把 densification importance 换成当前可实现的 Taming-style primitive/loss score 后，质量没有塌，平均还小幅上升。

更准确的解释是：FastGS 的 clone/split/prune schedule、scene overrides、final-prune tail 和优化流程本身已经提供了很强的泛化能力；VCD/VCP importance 更可能是容量效率和预算控制上的关键，而不是唯一质量来源。由于 0021 平均多出约 36.7% Gaussians，不能把质量提升解释为 Taming-style importance 本身更优。下一步若要隔离效率，需要做 matched-GS 或更强 prune/cap 对照。

重要口径：0021 是 `taming_importance_fastgs_prune`，只替换 densification 使用的 per-Gaussian importance；FastGS pruning score、clone/split gates、final-prune tail、scene overrides 和 densification interval=100 保持不变。它不是完整 Taming-3DGS densification 复现。

## Per-Scene Results

| scene | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | GS_num | dGS_num | train_time | dtrain_time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.4065 | +0.1497 | 0.770476 | +0.015210 | 0.216919 | -0.028065 | 2,090,511 | +530,302 | 197.48s | +38.37s |
| flowers | 21.6741 | +0.0456 | 0.606648 | +0.004317 | 0.333571 | -0.006817 | 1,481,180 | +358,365 | 173.30s | +27.62s |
| garden | 27.6643 | +0.0296 | 0.866539 | +0.002108 | 0.104985 | -0.004627 | 3,580,452 | +945,636 | 356.03s | +82.90s |
| stump | 27.2427 | +0.0643 | 0.793773 | +0.006927 | 0.223513 | -0.015819 | 1,476,951 | +412,091 | 162.78s | +31.77s |
| treehill | 22.9025 | +0.0750 | 0.642611 | +0.010290 | 0.351438 | -0.025515 | 1,337,254 | +328,144 | 157.45s | +30.05s |
| room | 32.2040 | -0.0096 | 0.933556 | +0.003110 | 0.178226 | -0.009927 | 821,293 | +249,686 | 139.44s | +25.92s |
| counter | 29.6121 | +0.0862 | 0.920692 | +0.002647 | 0.169076 | -0.007496 | 637,383 | +166,806 | 143.37s | +21.37s |
| kitchen | 32.2815 | +0.0005 | 0.939519 | +0.000564 | 0.102552 | -0.002556 | 1,434,403 | +256,415 | 265.07s | +45.22s |
| bonsai | 33.0503 | -0.0343 | 0.955130 | +0.001361 | 0.154316 | -0.005492 | 1,437,801 | +593,708 | 198.90s | +53.31s |

## Artifacts

Combined output:

```text
output/0021/taming_importance_fastgs_big_full9/mipnerf360_combined/summary.csv
output/0021/taming_importance_fastgs_big_full9/mipnerf360_combined/comparison_vs_fastgs_big_baseline.csv
output/0021/taming_importance_fastgs_big_full9/mipnerf360_combined/aggregate_vs_fastgs_big_baseline.csv
output/0021/taming_importance_fastgs_big_full9/mipnerf360_combined/averages.csv
```

Merge check:

```json
{
  "summary_rows": 9,
  "comparison_rows": 9,
  "expected_summary_rows": 9,
  "expected_comparison_rows": 9
}
```

## 启动记录

目标：验证 FastGS big 的 full9 指标优势是否来自 multi-view VCD/VCP importance。实验只替换 densification 使用的 per-Gaussian importance 为 Taming-3DGS-style score，FastGS pruning score、clone/split gates、final-prune tail、scene overrides 和 densification interval=100 保持不变。

启动：2026-05-26 11:58:58 CST 双卡 tmux full9。

完成：2026-05-26 12:25:50 CST 左右。原预估为 2026-05-26 12:35-12:50 CST；实际更快，merge 会话自动生成 combined summary。

启动前检查：

- `python -m py_compile` 通过。
- `bash -n scripts/run_0021_taming_importance_fastgs_big_tmux.sh` 通过。
- `git diff --check` 通过。
- `bonsai` 700-step smoke 通过，真实触发 densification step，并保存 checkpoint。smoke 最终 Gaussian 数为 311,622，训练时间 3.34s。

会话：

```text
0021_taming_imp_g0
0021_taming_imp_g1
0021_taming_imp_merge
```

场景分配：

```text
GPU0: garden bicycle flowers room
GPU1: kitchen bonsai stump treehill counter
```

输出位置：

```text
output/0021/taming_importance_fastgs_big_full9/mipnerf360_combined
```

raw logs：

```text
output/0021/debug_logs/0021_taming_imp_g0.log
output/0021/debug_logs/0021_taming_imp_g1.log
output/0021/debug_logs/0021_taming_imp_merge.log
```

对照 baseline：

```text
output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv
```

已汇总指标：

- PSNR
- SSIM
- LPIPS
- GS_num
- train_time
