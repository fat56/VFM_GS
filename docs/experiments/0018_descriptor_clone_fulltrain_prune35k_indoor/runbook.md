# 0018 运行手册

## 启动

```bash
bash scripts/run_0018_descriptor_clone_fulltrain_prune35k_indoor_tmux.sh start
```

tmux 会话：

```bash
tmux attach -t 0018_desc_clone_prune35_g0
tmux attach -t 0018_desc_clone_prune35_g1
tmux attach -t 0018_desc_clone_prune35_merge
```

日志：

```text
output/0018/debug_logs/0018_desc_clone_prune35_g0.log
output/0018/debug_logs/0018_desc_clone_prune35_g1.log
output/0018/debug_logs/0018_desc_clone_prune35_merge.log
```

GPU0 场景：

```text
room counter
```

GPU1 场景：

```text
kitchen bonsai
```

## 输出

双卡分组输出：

```text
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mip_g0
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mip_g1
```

合并输出：

```text
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/summary.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/comparison_20k_to_eval.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/aggregate_20k_to_eval.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/comparison_vs_baseline30.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/aggregate_vs_baseline30.csv
```

完整性预期：

```text
summary.csv: 20 行 = 4 个场景 x 5 个评测点
comparison_20k_to_eval.csv: 12 行 = 4 个场景 x 3 个 prune-tail 增量评测点
comparison_vs_baseline30.csv: 8 行 = 4 个场景 x 0018 的 30K/35K 对标准 baseline 30K
aggregate_20k_to_eval.csv: 3 行 = 相对 descriptor 20K 的 +5K / +10K / +15K
aggregate_vs_baseline30.csv: 2 行 = 0018 30K / 35K 对标准 FastGS baseline 30K
```

手动合并：

```bash
bash scripts/run_0018_descriptor_clone_fulltrain_prune35k_indoor_tmux.sh merge
```

状态检查：

```bash
tmux ls | grep 0018_desc_clone_prune35
tail -f output/0018/debug_logs/0018_desc_clone_prune35_g0.log
tail -f output/0018/debug_logs/0018_desc_clone_prune35_g1.log
```

## 单场景运行

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0018_descriptor_clone_fulltrain_prune35k_indoor_eval.py \
  --output-root output/0018/descriptor_clone_fulltrain_prune35k_indoor/manual \
  --scenes room
```

## 补充对照

0018 后续包含两个补充对照，不另开 0019；当前已完成：

- `rgb_fastgs_extra_fulltrain`：从 0 跑完整流程，15K-20K 使用 RGB/FastGS extra-densify，optimizer/final-prune schedule 与 0018 主实验一致。
- `desc16k21k_prune35k`：复用 0017 的 descriptor clone-only 21K PLY，从 21K 接 FastGS final-prune tail 到 35K，用来观察 PLY 续跑与从头完整训练的差异；该路径从 PLY 加载并重置 optimizer，不是严格的 optimizer-state 继承实验。

启动：

```bash
bash scripts/run_0018_supplemental_controls_tmux.sh start
```

tmux 会话：

```bash
tmux attach -t 0018_controls_g0
tmux attach -t 0018_controls_g1
tmux attach -t 0018_controls_merge
```

补充对照输出：

```text
output/0018/supplemental_controls/mip_g0
output/0018/supplemental_controls/mip_g1
output/0018/supplemental_controls/mipnerf360_indoor_combined/summary.csv
output/0018/supplemental_controls/mipnerf360_indoor_combined/comparison_start_to_eval.csv
output/0018/supplemental_controls/mipnerf360_indoor_combined/aggregate_start_to_eval.csv
output/0018/supplemental_controls/mipnerf360_indoor_combined/comparison_vs_baseline30.csv
output/0018/supplemental_controls/mipnerf360_indoor_combined/aggregate_vs_baseline30.csv
```

完整性预期：

```text
summary.csv: 36 行 = RGB/FastGS fulltrain 20 行 + 0017-21K continuation 16 行
comparison_start_to_eval.csv: 24 行 = 每个对照 4 场景 x 3 个后续评测点
comparison_vs_baseline30.csv: 16 行 = 2 个对照 x 4 场景 x 30K/35K
aggregate_start_to_eval.csv: 6 行 = 2 个对照 x 3 个后续评测点
aggregate_vs_baseline30.csv: 4 行 = 2 个对照 x 30K/35K
```

状态检查：

```bash
tmux ls | grep 0018_controls
tail -f output/0018/debug_logs/0018_controls_g0.log
tail -f output/0018/debug_logs/0018_controls_g1.log
```
