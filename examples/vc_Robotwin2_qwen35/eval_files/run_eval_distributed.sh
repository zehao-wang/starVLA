#!/bin/bash
# run_eval_distributed.sh — Full-node distributed eval for starVLA on RoboTwin2.
#
# One policy server per GPU, one eval worker per server.
# A shared task queue feeds all workers; each worker is always busy.
#
# Usage (from packages/starVLA/, starVLA env active):
#
#   bash examples/vc_Robotwin2_qwen35/eval_files/run_eval_distributed.sh \
#       <hf_dataset_name>     e.g. lerobot_robotwin_rand20k_debug
#       [exp_name]            default: starvla_dist_eval
#       [max_episodes]        default: none  (all episodes per task)
#       [gpu_ids]             default: auto-detect via nvidia-smi  (e.g. "0,1,2,3")
#       [base_port]           default: 5694  (ports = base, base+1, …)
#
# Environment variables (override defaults):
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
HF_DATASET_NAME=${1:-lerobot_robotwin_rand20k_debug}
EXP_NAME=${2:-starvla_dist_eval}
MAX_EPISODES=${3:-none}
GPU_IDS_ARG=${4:-}          # e.g. "0,1,2,3" — empty → auto-detect
BASE_PORT=${5:-5694}
SPLIT=${SPLIT:-Randomized}  # e.g. SPLIT=Clean  (env var, default: Randomized)

HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-RoboTwin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
ROBOTWIN_PATH=${ROBOTWIN_PATH:-$(realpath "$STARVLA_PATH/../../packages/RoboTwin")}
DEPLOY_POLICY_YML=${DEPLOY_POLICY_YML:-$EVAL_FILES_PATH/deploy_policy.yml}

# Read ckpt path from deploy_policy.yml
CKPT_PATH=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(open('$DEPLOY_POLICY_YML'))
ckpt = cfg.get('policy_ckpt_path', '')
if ckpt and not pathlib.Path(ckpt).is_absolute():
    ckpt = str(pathlib.Path('$STARVLA_PATH') / ckpt)
print(ckpt)
")

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: policy_ckpt_path not set in $DEPLOY_POLICY_YML"
    exit 1
fi

# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
if [[ -n "$GPU_IDS_ARG" ]]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_IDS_ARG"
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    GPU_IDS=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
fi

NUM_WORKERS=${#GPU_IDS[@]}

# Build comma-separated port list for the orchestrator
PORTS=""
SERVER_PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    PORTS="${PORTS:+$PORTS,}$PORT"
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  dataset       : $HF_DATASET_NAME"
echo "  exp_name      : $EXP_NAME"
echo "  max_episodes  : $MAX_EPISODES"
echo "  gpu_ids       : ${GPU_IDS[*]}"
echo "  base_port     : $BASE_PORT"
echo "  ports         : $PORTS"
echo "  num_workers   : $NUM_WORKERS"
echo "  policy_config : $DEPLOY_POLICY_YML"
echo "  ckpt          : $CKPT_PATH"
echo "  robotwin_path : $ROBOTWIN_PATH"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Start one policy server per GPU (background)
# ---------------------------------------------------------------------------
LOG_DIR=$EVAL_FILES_PATH/logs
mkdir -p "$LOG_DIR"

echo "[1/3] Starting $NUM_WORKERS policy server(s) …"
for ((i=0; i<NUM_WORKERS; i++)); do
    GPU_ID=${GPU_IDS[$i]}
    PORT=$((BASE_PORT + i))
    LOG=$LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log

    bash "$EVAL_FILES_PATH/run_policy_server_gpu.sh" \
        "$GPU_ID" "$PORT" "$CKPT_PATH" \
        > "$LOG" 2>&1 &
    SERVER_PIDS+=($!)
    echo "    GPU ${GPU_ID}  port ${PORT}  PID ${SERVER_PIDS[$i]}  log: $LOG"
done

# Cleanup trap — kill all servers on exit
cleanup() {
    echo ""
    echo "Stopping policy server(s) …"
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait until every server port is ready
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Waiting for all $NUM_WORKERS server(s) to be ready …"
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
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    elapsed=0
    while ! _port_open "$PORT"; do
        if [[ $elapsed -ge $MAX_WAIT ]]; then
            echo "ERROR: Server on port $PORT did not start within ${MAX_WAIT}s."
            echo "       Check $LOG_DIR/server_gpu${GPU_IDS[$i]}_port${PORT}.log"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "    port $PORT ready after ${elapsed}s"
done
echo ""

# ---------------------------------------------------------------------------
# Launch orchestrator (runs under RoboTwin conda env)
# ---------------------------------------------------------------------------
echo "[3/3] Launching eval orchestrator …"
echo ""

export PYTHONPATH=$ROBOTWIN_PATH:$STARVLA_PATH:$EVAL_FILES_PATH:${PYTHONPATH:-}
export HF_LEROBOT_HOME
export ROBOTWIN_PATH


GPU_IDS_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

conda run -n "$ROBOTWIN_CONDA_ENV" --no-capture-output \
    python "$EVAL_FILES_PATH/utils/eval_orchestrator.py" \
        --dataset-name  "$HF_DATASET_NAME" \
        --ports         "$PORTS" \
        --gpu-ids       "$GPU_IDS_CSV" \
        --exp-name      "$EXP_NAME" \
        --max-episodes  "$MAX_EPISODES" \
        --policy-config "$DEPLOY_POLICY_YML" \
        --split         "$SPLIT"

echo ""
echo "======================================================================"
echo "Distributed eval complete."
echo "======================================================================"
