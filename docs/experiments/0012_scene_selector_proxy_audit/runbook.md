# 0012 Runbook

Run:

```bash
uv run --active python scripts/run_0012_scene_selector_proxy_audit.py
```

Outputs:

```text
output/0012/scene_selector_proxy_audit/feature_table.csv
output/0012/scene_selector_proxy_audit/policy_summary.csv
output/0012/scene_selector_proxy_audit/loocv_predictions.csv
output/0012/scene_selector_proxy_audit/summary_stats.json
```

This is an offline audit over existing experiment outputs; it does not launch tmux training.
