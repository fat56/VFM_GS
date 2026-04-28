#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON:-python3}"

${PYTHON_BIN} -m compileall src/vfm_gs
${PYTHON_BIN} -m vfm_gs.cli.train --help >/dev/null
${PYTHON_BIN} -m vfm_gs.cli.render --help >/dev/null
${PYTHON_BIN} -m vfm_gs.cli.metrics --help >/dev/null
${PYTHON_BIN} -m vfm_gs.cli.convert --help >/dev/null
${PYTHON_BIN} -m vfm_gs.cli.build_vfm_cache --help >/dev/null
${PYTHON_BIN} -m vfm_gs.cli.full_eval --dry_run --mode big \
  --mipnerf360 /tmp/mipnerf360 \
  --tanksandtemples /tmp/tanksandtemples \
  --deepblending /tmp/deepblending \
  --output_path /tmp/vfm_gs_eval \
  --skip_metrics >/dev/null
