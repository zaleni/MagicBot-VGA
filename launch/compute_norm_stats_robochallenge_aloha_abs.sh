#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_TYPE="${ACTION_TYPE:-abs}"
if [[ "${ACTION_TYPE}" != "abs" ]]; then
  echo "This launcher is for ALOHA abs normalization stats. Got ACTION_TYPE=${ACTION_TYPE}."
  exit 1
fi

# Keep regular-only ALOHA by default, matching the abs finetune launcher.
export ROBOCHALLENGE_ALOHA_EXTRA_TASKS="${ROBOCHALLENGE_ALOHA_EXTRA_TASKS-}"

exec bash "${SCRIPT_DIR}/compute_norm_stats_robochallenge_aloha.sh"
