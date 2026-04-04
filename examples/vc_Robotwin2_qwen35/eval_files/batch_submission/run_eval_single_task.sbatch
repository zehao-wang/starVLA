#!/bin/bash
#SBATCH --job-name=starvla_eval_single_task
#SBATCH --output=examples/vc_Robotwin2_qwen35/eval_files/logs/slurm_%j.out
#SBATCH --error=examples/vc_Robotwin2_qwen35/eval_files/logs/slurm_%j.err
#SBATCH --exclusive
#SBATCH -p ondemand-g6e-1gpu-l40s
#SBATCH --time=12:00:00

set -e

# NOTE: Do not derive from $0 here.
# Under Slurm, the script can be staged to /var/spool/slurmd and $0 points there.
# We pass EVAL_FILES_DIR explicitly from submit_multi_tasks.sh.
EVAL_FILES_DIR=${EVAL_FILES_DIR:-}
if [[ -z "$EVAL_FILES_DIR" ]]; then
	echo "ERROR: EVAL_FILES_DIR is empty. Submit via batch_submission/submit_multi_tasks.sh."
	exit 1
fi
EVAL_FILES_DIR=$(realpath "$EVAL_FILES_DIR")
if [[ ! -f "$EVAL_FILES_DIR/run_eval_single_task.sh" ]]; then
	echo "ERROR: invalid EVAL_FILES_DIR: $EVAL_FILES_DIR"
	exit 1
fi

# ---------------------------------------------------------------------------
# External parameters (override via: sbatch --export=ALL,KEY=VALUE,...)
# ---------------------------------------------------------------------------
HF_DATASET_NAME=${HF_DATASET_NAME:-lerobot_robotwin_rand20k_debug}
TASK_NAME=${TASK_NAME:-adjust_bottle}
MAX_EPISODES=${MAX_EPISODES:-none}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5694}

DEPLOY_POLICY_YML=${DEPLOY_POLICY_YML:-$EVAL_FILES_DIR/deploy_policy.yml}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-RoboTwin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-$(realpath "$EVAL_FILES_DIR/../../../../RoboTwin")}
SPLIT=${SPLIT:-Randomized}

if [[ ! -f "$DEPLOY_POLICY_YML" ]]; then
	echo "ERROR: DEPLOY_POLICY_YML not found: $DEPLOY_POLICY_YML"
	exit 1
fi
DEPLOY_POLICY_YML=$(realpath "$DEPLOY_POLICY_YML")

mkdir -p "$EVAL_FILES_DIR/logs"

echo "======================================================================"
echo "SLURM_JOB_ID    : ${SLURM_JOB_ID:-local}"
echo "dataset         : $HF_DATASET_NAME"
echo "task_name       : $TASK_NAME"
echo "max_episodes    : $MAX_EPISODES"
echo "gpu_id          : $GPU_ID"
echo "port            : $PORT"
echo "split           : $SPLIT"
echo "deploy_yml      : $DEPLOY_POLICY_YML"
echo "hf_lerobot_home : $HF_LEROBOT_HOME"
echo "robotwin_env    : $ROBOTWIN_CONDA_ENV"
echo "starvla_python  : $STAR_VLA_PYTHON"
echo "robotwin_path   : $ROBOTWIN_PATH"
echo "======================================================================"

bash "$EVAL_FILES_DIR/run_eval_single_task.sh"
