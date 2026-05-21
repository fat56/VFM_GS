# 0011 Runbook

## Start

```bash
bash scripts/run_0011_descriptor_i050_until4000_tmux.sh start
```

Sessions:

```bash
tmux attach -t 0011_i050_u4000_g0
tmux attach -t 0011_i050_u4000_g1
tmux attach -t 0011_i050_u4000_merge
```

Logs:

```text
output/0011/debug_logs/0011_i050_u4000_g0.log
output/0011/debug_logs/0011_i050_u4000_g1.log
output/0011/debug_logs/0011_i050_u4000_merge.log
```

GPU0 scenes:

```text
bicycle flowers garden stump treehill
```

GPU1 scenes:

```text
room counter kitchen bonsai
```

Merged outputs:

```text
output/0011/descriptor_i050_until4000/mipnerf360_combined/summary.csv
output/0011/descriptor_i050_until4000/mipnerf360_combined/comparison_vs_phase0.csv
output/0011/descriptor_i050_until4000/mipnerf360_combined/comparison_vs_until8000.csv
output/0011/descriptor_i050_until4000/mipnerf360_combined/summary_stats.json
```
