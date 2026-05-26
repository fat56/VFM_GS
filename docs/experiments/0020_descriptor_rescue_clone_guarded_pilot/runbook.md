# 0020 Runbook

## 启动

```bash
bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh start
```

默认 tmux 会话：

```text
0020_desc_rescue_g0
0020_desc_rescue_g1
0020_desc_rescue_merge
```

默认场景分配：

```text
GPU0: garden room
GPU1: stump treehill
```

可以用环境变量覆盖：

```bash
SCENES_G0="garden" SCENES_G1="room stump treehill" \
  bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh start
```

## 监控

```bash
tmux ls | grep 0020_desc_rescue
tmux attach -t 0020_desc_rescue_g0
tmux attach -t 0020_desc_rescue_g1
tmux attach -t 0020_desc_rescue_merge
```

日志：

```text
output/0020/debug_logs/0020_desc_rescue_g0.log
output/0020/debug_logs/0020_desc_rescue_g1.log
output/0020/debug_logs/0020_desc_rescue_merge.log
```

## 输出

```text
output/0020/descriptor_rescue_clone_guarded_pilot/mip_g0
output/0020/descriptor_rescue_clone_guarded_pilot/mip_g1
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined
```

主要合并表：

```text
summary.csv
comparison_15k_to_eval.csv
aggregate_15k_to_eval.csv
comparison_vs_baseline_curve.csv
aggregate_vs_baseline_curve.csv
comparison_vs_baseline30.csv
aggregate_vs_baseline30.csv
```

## 手动合并

```bash
bash scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh merge
```

## 判定

优先看 30K 相对 FastGS 30K：

- `room/garden` 是否从 0019 的明显负向中恢复。
- `stump/treehill` 是否保留 SSIM/LPIPS 正向。
- 4 场景平均是否至少不牺牲 PSNR，同时维持 LPIPS/SSIM 收益。
