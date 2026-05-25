# 0019 运行手册

## 启动

```bash
bash scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_tmux.sh start
```

tmux 会话：

```bash
tmux attach -t 0019_desc_or_rgb_clone_g0
tmux attach -t 0019_desc_or_rgb_clone_g1
tmux attach -t 0019_desc_or_rgb_clone_merge
```

日志：

```text
output/0019/debug_logs/0019_desc_or_rgb_clone_g0.log
output/0019/debug_logs/0019_desc_or_rgb_clone_g1.log
output/0019/debug_logs/0019_desc_or_rgb_clone_merge.log
```

GPU0 场景：

```text
garden bicycle flowers room
```

GPU1 场景：

```text
kitchen bonsai stump treehill counter
```

该分配按既有 FastGS 30K checkpoint-curve 训练时长粗略均衡：GPU0 约 689s baseline 训练量，GPU1 约 743s baseline 训练量。0019 还会在 0-15K 额外计算 descriptor clone 候选，因此实际耗时以日志为准。

## 输出

双卡分组输出：

```text
output/0019/descriptor_or_rgb_clone_baseline30_full9/mip_g0
output/0019/descriptor_or_rgb_clone_baseline30_full9/mip_g1
```

合并输出：

```text
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/summary.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/comparison_15k_to_eval.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/aggregate_15k_to_eval.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/comparison_vs_baseline_curve.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/aggregate_vs_baseline_curve.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/comparison_vs_baseline30.csv
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined/aggregate_vs_baseline30.csv
```

完整性预期：

```text
summary.csv: 36 行 = 9 个场景 x 4 个评测点
comparison_15k_to_eval.csv: 27 行 = 9 个场景 x 20K/25K/30K 相对 15K
comparison_vs_baseline_curve.csv: 18 行 = 9 个场景 x baseline curve 中有精确对齐的 20K/30K
comparison_vs_baseline30.csv: 9 行 = 9 个场景 x 0019 30K 对 FastGS 30K
aggregate_15k_to_eval.csv: 3 行
aggregate_vs_baseline_curve.csv: 2 行
aggregate_vs_baseline30.csv: 1 行
```

手动合并：

```bash
bash scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_tmux.sh merge
```

状态检查：

```bash
tmux ls | grep 0019_desc_or_rgb_clone
tail -f output/0019/debug_logs/0019_desc_or_rgb_clone_g0.log
tail -f output/0019/debug_logs/0019_desc_or_rgb_clone_g1.log
```

## 单场景运行

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py \
  --output-root output/0019/descriptor_or_rgb_clone_baseline30_full9/manual \
  --scenes bicycle
```
