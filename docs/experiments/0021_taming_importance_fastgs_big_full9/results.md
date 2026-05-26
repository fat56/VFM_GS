# 0021 Results

## 启动记录

目标：验证 FastGS big 的 full9 指标优势是否来自 multi-view VCD/VCP importance。实验只替换 densification 使用的 per-Gaussian importance 为 Taming-3DGS-style score，FastGS pruning score、clone/split gates、final-prune tail、scene overrides 和 densification interval=100 保持不变。

启动：2026-05-26 11:58:58 CST 双卡 tmux full9。

预估完成：2026-05-26 12:35-12:50 CST。该估计基于 0002/0019/0020 的 full9 历史耗时、本实验无 DINO cache 但 scorer 每个采样视角多一次 Taming saliency count render 的开销；render/metrics 完成后 merge 会话会自动合并。

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

当前 raw logs：

```text
output/0021/debug_logs/0021_taming_imp_g0.log
output/0021/debug_logs/0021_taming_imp_g1.log
output/0021/debug_logs/0021_taming_imp_merge.log
```

对照 baseline：

```text
output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv
```

待汇总指标：

- PSNR
- SSIM
- LPIPS
- GS_num
- train_time
