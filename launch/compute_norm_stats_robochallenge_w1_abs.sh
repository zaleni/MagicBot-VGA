#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_TYPE="${ACTION_TYPE:-abs}"
if [[ "${ACTION_TYPE}" != "abs" ]]; then
  echo "This launcher is for W1 abs normalization stats. Got ACTION_TYPE=${ACTION_TYPE}."
  exit 1
fi

# Keep regular-only W1 by default, matching cubev2_finetune_robochallenge_w1_abs_from_delta70k.sh.
export ROBOCHALLENGE_W1_EXTRA_TASKS="${ROBOCHALLENGE_W1_EXTRA_TASKS-}"

exec bash "${SCRIPT_DIR}/compute_norm_stats_robochallenge_w1.sh"
