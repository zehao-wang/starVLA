#!/bin/bash
# Submit one single-GPU sbatch job per SimplerEnv task.
#
# Usage from packages/starVLA:
#   CKPT_PATH=results/Checkpoints/.../final_model/pytorch_model.pt \
#   bash examples/vc_simplerenv_qwen35/eval_files/batch_submission/submit_multi_tasks.sh
#
# Or indirectly via the generated run_eval.sh in the checkpoint directory.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EVAL_FILES_DIR=$(realpath "$SCRIPT_DIR/..")
SBATCH_SCRIPT="$SCRIPT_DIR/run_eval_single_task.sbatch"
REPO_ROOT=${REPO_ROOT:-$(realpath "$EVAL_FILES_DIR/../../..")}

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

CKPT_PATH=${CKPT_PATH:-}
PORT=${PORT:-6780}
OBJ_EPISODE_START=${OBJ_EPISODE_START:-0}
OBJ_EPISODE_END=${OBJ_EPISODE_END:-24}
PREFLIGHT_SIM=${PREFLIGHT_SIM:-0}
EXP_NAME=${EXP_NAME:-}
DATASET_NAME=${DATASET_NAME:-bridge_rt_1}

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/shared/home/ZWA0839/.conda/envs/simpler_env/bin/python}
SIMPLERENV_PATH=${SIMPLERENV_PATH:-/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv}

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: CKPT_PATH is empty."
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Edit this list to control which jobs are submitted.
TASK_LIST=(
    StackGreenCubeOnYellowCubeBakedTexInScene-v0
    PutCarrotOnPlateInScene-v0
    PutSpoonOnTableClothInScene-v0
    PutEggplantInBasketScene-v0
)

mkdir -p "$EVAL_FILES_DIR/logs"

echo "======================================================================"
echo "Submitting ${#TASK_LIST[@]} SimplerEnv task job(s)"
echo "ckpt_path       : $CKPT_PATH"
echo "port            : $PORT"
echo "obj episodes    : [$OBJ_EPISODE_START, $OBJ_EPISODE_END)"
echo "preflight_sim   : $PREFLIGHT_SIM"
echo "eval_files_dir  : $EVAL_FILES_DIR"
echo "======================================================================"

for task in "${TASK_LIST[@]}"; do
    echo "[submit] task=$task"
    sbatch \
        --job-name="se_${task}" \
        --export=ALL,\
TASK_NAME="$task",\
CKPT_PATH="$CKPT_PATH",\
PORT="$PORT",\
OBJ_EPISODE_START="$OBJ_EPISODE_START",\
OBJ_EPISODE_END="$OBJ_EPISODE_END",\
PREFLIGHT_SIM="$PREFLIGHT_SIM",\
EXP_NAME="$EXP_NAME",\
DATASET_NAME="$DATASET_NAME",\
REPO_ROOT="$REPO_ROOT",\
STAR_VLA_PYTHON="$STAR_VLA_PYTHON",\
SIM_PYTHON="$SIM_PYTHON",\
SIMPLERENV_PATH="$SIMPLERENV_PATH",\
EVAL_FILES_DIR="$EVAL_FILES_DIR" \
        "$SBATCH_SCRIPT"
done

echo "Done."
