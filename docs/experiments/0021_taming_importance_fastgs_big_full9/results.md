# 0021 Results

## 启动记录

目标：验证 FastGS big 的 full9 指标优势是否来自 multi-view VCD/VCP importance。实验只替换 densification 使用的 per-Gaussian importance 为 Taming-3DGS-style score，FastGS pruning score、clone/split gates、final-prune tail、scene overrides 和 densification interval=100 保持不变。

状态：配置、脚本与 scorer 已准备，等待启动双卡 tmux full9。

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
