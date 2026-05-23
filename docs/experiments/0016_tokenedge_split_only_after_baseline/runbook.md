# 0016 Runbook

## Start

```bash
bash scripts/run_0016_tokenedge_split_only_tmux.sh start
```

## Monitor

```bash
tmux ls | grep 0016_te_split
tail -f output/0016/debug_logs/0016_te_split_g0.log
tail -f output/0016/debug_logs/0016_te_split_g1.log
```

## Merge Manually

```bash
bash scripts/run_0016_tokenedge_split_only_tmux.sh merge
```

## Key Files

- `output/0016/tokenedge_split_only_after_baseline/mipnerf360_combined/summary.csv`
- `output/0016/tokenedge_split_only_after_baseline/mipnerf360_combined/comparison_start_to_final.csv`
- `output/0016/tokenedge_split_only_after_baseline/mipnerf360_combined/aggregate_by_switch.csv`

## Supplemental Smoke

Start:

```bash
bash scripts/run_0016_supplemental_smoke_tmux.sh start
```

Monitor:

```bash
tmux ls | grep 0016_smoke
tail -f output/0016/debug_logs/0016_smoke_g0.log
tail -f output/0016/debug_logs/0016_smoke_g1.log
```

Merge manually:

```bash
bash scripts/run_0016_supplemental_smoke_tmux.sh merge
```

Key files:

- `output/0016/supplemental_smoke/mipnerf360_smoke_combined/summary.csv`
- `output/0016/supplemental_smoke/mipnerf360_smoke_combined/comparison_start_to_eval.csv`
- `output/0016/supplemental_smoke/mipnerf360_smoke_combined/aggregate_by_variant.csv`

## Notes

The runner warm-starts from baseline checkpoint-curve PLY files, not optimizer checkpoints. This keeps the experiment cheap and uses the available 16K/18K/20K baseline states, but optimizer momentum is restarted for the 5K continuation.
