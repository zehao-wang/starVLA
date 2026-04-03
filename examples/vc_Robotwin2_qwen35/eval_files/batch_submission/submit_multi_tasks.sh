#!/bin/bash
# submit_multi_tasks.sh — Submit one sbatch job per task.
#
# Usage examples:
#   bash .../batch_submission/submit_multi_tasks.sh
#
# Edit TASK_LIST below to control which tasks are submitted.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SBATCH_SCRIPT="$SCRIPT_DIR/run_eval_single_task.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

# ---------------------------------------------------------------------------
# External parameters (shared defaults)
# ---------------------------------------------------------------------------
HF_DATASET_NAME=${HF_DATASET_NAME:-lerobot_robotwin_rand20k_debug}
MAX_EPISODES=${MAX_EPISODES:-none}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5694}

DEPLOY_POLICY_YML=${DEPLOY_POLICY_YML:-$SCRIPT_DIR/../deploy_policy.yml}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-RoboTwin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-$(realpath "$SCRIPT_DIR/../../../../RoboTwin")}
SPLIT=${SPLIT:-Randomized}

# ---------------------------------------------------------------------------
# Task list (edit this section)
# ---------------------------------------------------------------------------
TASK_LIST=(
    "adjust_bottle"
    # "open_box"
    # "stack_block"
)

if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no valid task names found."
    exit 1
fi

echo "======================================================================"
echo "Submitting ${#TASK_LIST[@]} task(s) with one sbatch per task"
echo "dataset         : $HF_DATASET_NAME"
echo "max_episodes    : $MAX_EPISODES"
echo "gpu_id          : $GPU_ID"
echo "port            : $PORT"
echo "split           : $SPLIT"
echo "deploy_yml      : $DEPLOY_POLICY_YML"
echo "======================================================================"

for task in "${TASK_LIST[@]}"; do
    echo "[submit] task=$task"
    sbatch \
        --job-name="eval_${task}" \
        --export=ALL,\
TASK_NAME="$task",\
HF_DATASET_NAME="$HF_DATASET_NAME",\
MAX_EPISODES="$MAX_EPISODES",\
GPU_ID="$GPU_ID",\
PORT="$PORT",\
DEPLOY_POLICY_YML="$DEPLOY_POLICY_YML",\
HF_LEROBOT_HOME="$HF_LEROBOT_HOME",\
ROBOTWIN_CONDA_ENV="$ROBOTWIN_CONDA_ENV",\
STAR_VLA_PYTHON="$STAR_VLA_PYTHON",\
ROBOTWIN_PATH="$ROBOTWIN_PATH",\
SPLIT="$SPLIT" \
        "$SBATCH_SCRIPT"
done

echo "Done."
