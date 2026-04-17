#!/bin/bash
#SBATCH --job-name=simplerenv_eval_single_task
#SBATCH --output=examples/vc_simplerenv_qwen35/eval_files/logs/slurm_%j.out
#SBATCH --error=examples/vc_simplerenv_qwen35/eval_files/logs/slurm_%j.err
#SBATCH -p ondemand-g6e-1gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -e

# Submit via submit_multi_tasks.sh so EVAL_FILES_DIR is absolute and stable.
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

TASK_NAME=${TASK_NAME:-}
CKPT_PATH=${CKPT_PATH:-}
PORT=${PORT:-6780}
OBJ_EPISODE_START=${OBJ_EPISODE_START:-0}
OBJ_EPISODE_END=${OBJ_EPISODE_END:-24}
PREFLIGHT_SIM=${PREFLIGHT_SIM:-0}

REPO_ROOT=${REPO_ROOT:-$(realpath "$EVAL_FILES_DIR/../../..")}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/shared/home/ZWA0839/.conda/envs/simpler_env/bin/python}
SIMPLERENV_PATH=${SIMPLERENV_PATH:-/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv}

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: CKPT_PATH is empty."
    exit 1
fi

mkdir -p "$EVAL_FILES_DIR/logs"

echo "======================================================================"
echo "SLURM_JOB_ID      : ${SLURM_JOB_ID:-local}"
echo "task_name         : $TASK_NAME"
echo "ckpt_path         : $CKPT_PATH"
echo "port              : $PORT"
echo "obj episodes      : [$OBJ_EPISODE_START, $OBJ_EPISODE_END)"
echo "preflight_sim     : $PREFLIGHT_SIM"
echo "cuda_visible      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "eval_files_dir    : $EVAL_FILES_DIR"
echo "repo_root         : $REPO_ROOT"
echo "star_vla_python   : $STAR_VLA_PYTHON"
echo "sim_python        : $SIM_PYTHON"
echo "simplerenv_path   : $SIMPLERENV_PATH"
echo "======================================================================"

bash "$EVAL_FILES_DIR/run_eval_single_task.sh"
