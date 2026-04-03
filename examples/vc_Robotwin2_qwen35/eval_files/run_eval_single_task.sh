#!/bin/bash
# run_eval_single_task.sh — Single-GPU eval for one specified task on RoboTwin2.
#
# One policy server + one eval worker.
#
# Usage (from packages/starVLA/, starVLA env active):
#
#   # Recommended (unified): env vars + optional --flags
#   HF_DATASET_NAME=lerobot_robotwin_rand20k_debug TASK_NAME=adjust_bottle \
#   bash examples/vc_Robotwin2_qwen35/eval_files/run_eval_single_task.sh
#
#   # Optional CLI overrides (same names as env vars)
#   bash examples/vc_Robotwin2_qwen35/eval_files/run_eval_single_task.sh \
#       --dataset-name lerobot_robotwin_rand20k_debug \
#       --task-name adjust_bottle --gpu-id 0 --port 5694
#
# exp_name is read from DEPLOY_POLICY_YML (field: exp_name).
# If missing/empty, fallback: starvla_single_task_eval
#
# Environment variables (override defaults):
#   HF_DATASET_NAME      lerobot dataset name
#   TASK_NAME            eval task name  (default: adjust_bottle)
#   MAX_EPISODES         max episodes per task  (default: none)
#   GPU_ID               GPU ID  (default: 0)
#   PORT                 policy server port  (default: 5694)
#   DEPLOY_POLICY_YML    path to deploy_policy.yml  (default: eval_files/deploy_policy.yml)
#   HF_LEROBOT_HOME      path to lerobot dataset root
#   ROBOTWIN_CONDA_ENV   name of the RoboTwin conda environment
#   STAR_VLA_PYTHON      path to the starVLA python binary
#   ROBOTWIN_PATH        path to packages/RoboTwin  (auto-derived if unset)
#   SPLIT                dataset split subdirectory: Clean or Randomized  (default: Randomized)

set -e

EVAL_FILES_PATH=$(cd "$(dirname "$0")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_DATASET_NAME=${HF_DATASET_NAME:-lerobot_robotwin_rand20k_debug}
TASK_NAME=${TASK_NAME:-adjust_bottle}
MAX_EPISODES=${MAX_EPISODES:-none}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5694}
SPLIT=${SPLIT:-Randomized}

HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-RoboTwin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}

print_usage() {
    cat <<'EOF'
Usage:
  run_eval_single_task.sh [--dataset-name NAME] [--task-name TASK]
                          [--max-episodes N|none] [--gpu-id ID] [--port PORT]
                          [--split SPLIT] [--deploy-policy-yml PATH]
                          [--hf-lerobot-home PATH] [--robotwin-conda-env NAME]
                          [--star-vla-python PATH] [--robotwin-path PATH]
EOF
}

# Unified long options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name) HF_DATASET_NAME="$2"; shift 2 ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --deploy-policy-yml) DEPLOY_POLICY_YML="$2"; shift 2 ;;
        --hf-lerobot-home) HF_LEROBOT_HOME="$2"; shift 2 ;;
        --robotwin-conda-env) ROBOTWIN_CONDA_ENV="$2"; shift 2 ;;
        --star-vla-python) STAR_VLA_PYTHON="$2"; shift 2 ;;
        --robotwin-path) ROBOTWIN_PATH="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; print_usage; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
ROBOTWIN_PATH=${ROBOTWIN_PATH:-$(realpath "$STARVLA_PATH/../../packages/RoboTwin")}
DEPLOY_POLICY_YML=${DEPLOY_POLICY_YML:-$EVAL_FILES_PATH/deploy_policy.yml}

if [[ ! -f "$DEPLOY_POLICY_YML" ]]; then
    echo "ERROR: DEPLOY_POLICY_YML not found: $DEPLOY_POLICY_YML"
    exit 1
fi
DEPLOY_POLICY_YML=$(realpath "$DEPLOY_POLICY_YML")

if [[ -z "$TASK_NAME" ]]; then
    echo "ERROR: TASK_NAME is empty. Use env TASK_NAME or --task-name."
    exit 1
fi

# Read ckpt path and exp name from deploy_policy.yml
readarray -t _POLICY_INFO < <(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(open('$DEPLOY_POLICY_YML'))
ckpt = cfg.get('policy_ckpt_path', '')
exp_name = cfg.get('exp_name', '')
if ckpt and not pathlib.Path(ckpt).is_absolute():
    ckpt = str(pathlib.Path('$STARVLA_PATH') / ckpt)
print(ckpt)
print(exp_name)
")
CKPT_PATH=${_POLICY_INFO[0]}
EXP_NAME=${_POLICY_INFO[1]:-starvla_single_task_eval}

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: policy_ckpt_path not set in $DEPLOY_POLICY_YML"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME=starvla_single_task_eval
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  dataset       : $HF_DATASET_NAME"
echo "  task_name     : $TASK_NAME"
echo "  exp_name      : $EXP_NAME"
echo "  max_episodes  : $MAX_EPISODES"
echo "  gpu_id        : $GPU_ID"
echo "  port          : $PORT"
echo "  split         : $SPLIT"
echo "  policy_config : $DEPLOY_POLICY_YML"
echo "  ckpt          : $CKPT_PATH"
echo "  robotwin_path : $ROBOTWIN_PATH"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Start policy server (background)
# ---------------------------------------------------------------------------
LOG_DIR=$EVAL_FILES_PATH/logs
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log

echo "[1/3] Starting policy server ..."
bash "$EVAL_FILES_PATH/run_policy_server_gpu.sh" \
    "$GPU_ID" "$PORT" "$CKPT_PATH" \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "    GPU ${GPU_ID}  port ${PORT}  PID ${SERVER_PID}  log: $LOG"

# Cleanup trap
cleanup() {
    echo ""
    echo "Stopping policy server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait until server port is ready
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Waiting for server to be ready ..."
MAX_WAIT=360
_port_open() {
    python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', int(sys.argv[1])))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" "$1" 2>/dev/null
}

elapsed=0
while ! _port_open "$PORT"; do
    if [[ $elapsed -ge $MAX_WAIT ]]; then
        echo "ERROR: Server on port $PORT did not start within ${MAX_WAIT}s."
        echo "       Check $LOG"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "    port $PORT ready after ${elapsed}s"
echo ""

# ---------------------------------------------------------------------------
# Launch orchestrator (runs under RoboTwin conda env)
# ---------------------------------------------------------------------------
echo "[3/3] Launching single-task eval orchestrator ..."
echo ""

export PYTHONPATH=$ROBOTWIN_PATH:$STARVLA_PATH:$EVAL_FILES_PATH:${PYTHONPATH:-}
export HF_LEROBOT_HOME
export ROBOTWIN_PATH

conda run -n "$ROBOTWIN_CONDA_ENV" --no-capture-output \
    python "$EVAL_FILES_PATH/utils/eval_orchestrator.py" \
        --dataset-name  "$HF_DATASET_NAME" \
        --task-name     "$TASK_NAME" \
        --ports         "$PORT" \
        --gpu-ids       "$GPU_ID" \
        --exp-name      "$EXP_NAME" \
        --max-episodes  "$MAX_EPISODES" \
        --policy-config "$DEPLOY_POLICY_YML" \
        --split         "$SPLIT"

echo ""
echo "======================================================================"
echo "Single-task eval complete."
echo "======================================================================"
