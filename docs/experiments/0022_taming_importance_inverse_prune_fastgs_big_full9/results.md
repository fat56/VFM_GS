# 0022 Results

## 启动记录

目标：在 0021 的 Taming-style clone/split importance 基础上，把 densify-stage prune 的采样权重改为 inverse Taming-style importance，验证是否能减少 0021 的 Gaussian 膨胀并保留质量收益。FastGS final-prune tail、scene overrides、clone/split gates 和 densification interval=100 保持不变。

启动：2026-05-26 13:15:14 CST 双卡 tmux full9。

状态：运行中。13:17 CST 检查时，`garden` 与 `kitchen` 首场景均已完成 camera load 并进入 30K training；`train.log` 中分别约到 6.7K / 7.8K iterations，已越过 500-step densification start，说明 inverse-importance densify-stage prune 路径已进入正式长跑。两张 RTX 5090 均有 `.venv/bin/python3` 训练进程，GPU util 约 92% / 96%。

预计：按 0021 同口径 full9 实际用时约 27 分钟、且 0022 首场景当前速度正常，预计 combined summary 会在 2026-05-26 13:45-14:00 CST 左右生成。

启动前检查：

- `python -m py_compile` 通过。
- `bash -n scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh` 通过。
- `git diff --check` 通过。
- `bonsai` 700-step smoke 通过，`cfg_args` 确认 `densify_prune_score_source='importance_inverse'`，保存 `point_cloud/iteration_700/point_cloud.ply`。当前 artifact PLY header 为 250,156 vertices。

会话：

```text
0022_taming_invprune_g0
0022_taming_invprune_g1
0022_taming_invprune_merge
```

场景分配：

```text
GPU0: garden bicycle flowers room
GPU1: kitchen bonsai stump treehill counter
```

对照：

- 0002 `fastgs_big` full9 baseline。
- 0021 `taming_importance_fastgs_prune_densify100`。

输出位置：

```text
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mipnerf360_combined
```

raw logs：

```text
output/0022/debug_logs/0022_taming_invprune_g0.log
output/0022/debug_logs/0022_taming_invprune_g1.log
output/0022/debug_logs/0022_taming_invprune_merge.log
```

待汇总指标：

- PSNR
- SSIM
- LPIPS
- GS_num
- train_time
