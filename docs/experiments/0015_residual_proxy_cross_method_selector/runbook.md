# 0015 Runbook

Run:

```bash
uv run --active python scripts/run_0015_residual_proxy_cross_method_selector.py
```

Outputs:

```text
output/0015/residual_proxy_cross_method_selector/feature_table.csv
output/0015/residual_proxy_cross_method_selector/loocv_predictions.csv
output/0015/residual_proxy_cross_method_selector/policy_summary.csv
output/0015/residual_proxy_cross_method_selector/summary_stats.json
```

This is an offline selector audit over existing 0008 and 0013 artifacts. It does not launch tmux training.
