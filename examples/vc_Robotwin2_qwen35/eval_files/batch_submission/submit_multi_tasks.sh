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
EVAL_FILES_DIR=$(realpath "$SCRIPT_DIR/..")

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
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch"
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
echo "eval_files_dir  : $EVAL_FILES_DIR"
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
EVAL_FILES_DIR="$EVAL_FILES_DIR",\
SPLIT="$SPLIT" \
        "$SBATCH_SCRIPT"
done

echo "Done."
