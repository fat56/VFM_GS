# 0017 Runbook

## Start

```bash
bash scripts/run_0017_descriptor_clone16k_full9_tmux.sh start
```

Sessions:

```bash
tmux attach -t 0017_desc_clone16k_g0
tmux attach -t 0017_desc_clone16k_g1
tmux attach -t 0017_desc_clone16k_merge
```

Logs:

```text
output/0017/debug_logs/0017_desc_clone16k_g0.log
output/0017/debug_logs/0017_desc_clone16k_g1.log
output/0017/debug_logs/0017_desc_clone16k_merge.log
```

GPU0 scenes:

```text
bicycle flowers garden stump treehill
```

GPU1 scenes:

```text
room counter kitchen bonsai
```

## Outputs

Per-GPU outputs:

```text
output/0017/descriptor_clone16k_full9/mip_g0
output/0017/descriptor_clone16k_full9/mip_g1
```

Merged outputs:

```text
output/0017/descriptor_clone16k_full9/mipnerf360_combined/summary.csv
output/0017/descriptor_clone16k_full9/mipnerf360_combined/comparison_start_to_eval.csv
output/0017/descriptor_clone16k_full9/mipnerf360_combined/aggregate_by_iteration.csv
```

Expected completeness:

```text
summary.csv: 54 rows = 9 scenes x 6 eval points
comparison_start_to_eval.csv: 45 rows = 9 scenes x 5 eval deltas
aggregate_by_iteration.csv: 5 rows = +1K / +2K / +3K / +4K / +5K
```

Manual merge:

```bash
bash scripts/run_0017_descriptor_clone16k_full9_tmux.sh merge
```

Status check:

```bash
tmux ls | grep 0017_desc_clone16k
tail -f output/0017/debug_logs/0017_desc_clone16k_g0.log
tail -f output/0017/debug_logs/0017_desc_clone16k_g1.log
```
