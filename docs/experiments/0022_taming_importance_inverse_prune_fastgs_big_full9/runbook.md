# 0022 Runbook

## Files

```text
configs/experiments/0022_taming_importance_inverse_prune_fastgs_big_full9.yaml
scripts/run_0022_taming_importance_inverse_prune_fastgs_big_eval.py
scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh
```

## Smoke / Static Checks

```bash
python -m py_compile \
  src/vfm_gs/scene/gaussian_model.py \
  src/vfm_gs/config/legacy_args.py \
  src/vfm_gs/scorers/taming_importance.py \
  scripts/run_0022_taming_importance_inverse_prune_fastgs_big_eval.py

bash -n scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh
git diff --check
```

## Full9 Launch

```bash
scripts/run_0022_taming_importance_inverse_prune_fastgs_big_tmux.sh start
```

Default split:

```text
GPU0: garden bicycle flowers room
GPU1: kitchen bonsai stump treehill counter
```

Sessions:

```text
0022_taming_invprune_g0
0022_taming_invprune_g1
0022_taming_invprune_merge
```

## Outputs

Per GPU:

```text
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mip_g0
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mip_g1
```

Combined:

```text
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mipnerf360_combined/summary.csv
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mipnerf360_combined/comparison_vs_fastgs_big_baseline.csv
output/0022/taming_importance_inverse_prune_fastgs_big_full9/mipnerf360_combined/aggregate_vs_fastgs_big_baseline.csv
```

Logs:

```text
output/0022/debug_logs/0022_taming_invprune_g0.log
output/0022/debug_logs/0022_taming_invprune_g1.log
output/0022/debug_logs/0022_taming_invprune_merge.log
```
