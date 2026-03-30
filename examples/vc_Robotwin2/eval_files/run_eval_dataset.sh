#!/bin/bash
# run_eval_dataset.sh — Start policy server + run dataset-mode eval for all tasks.
#
# Usage:
#   bash examples/vc_Robotwin2/eval_files/run_eval_dataset.sh \
#       <hf_dataset_name>   e.g. lerobot_robotwin_rand20k_debug
#       [exp_name]          default: starvla_dataset_eval
#       [max_episodes]      default: none (all)
#       [gpu_id]            default: 0
#
# Expects to be run from packages/starVLA/ with the starVLA conda env active.
# Eval client runs under the RoboTwin conda env (via conda run).

set -e

EVAL_FILES_PATH=$(cd "$(dirname "$0")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..

# ---------------------------------------------------------------------------
# Config (edit here or pass as args)
# ---------------------------------------------------------------------------
HF_DATASET_NAME=${1:-lerobot_robotwin_rand20k_debug}
EXP_NAME=${2:-starvla_dataset_eval}
MAX_EPISODES=${3:-none}
GPU_ID=${4:-0}

HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-RoboTwin}
SERVER_PORT=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$EVAL_FILES_PATH/deploy_policy.yml')); print(cfg.get('port', 5694))")

SERVER_LOG=$EVAL_FILES_PATH/server.log

# ---------------------------------------------------------------------------
# Discover tasks (subdirs of HF_LEROBOT_HOME/<dataset>/Randomized/)
# ---------------------------------------------------------------------------
DATASET_ROOT=$HF_LEROBOT_HOME/$HF_DATASET_NAME/Randomized
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: Dataset not found: $DATASET_ROOT"
    exit 1
fi

TASKS=()
for task_dir in "$DATASET_ROOT"/*/; do
    task_name=$(basename "$task_dir")
    TASKS+=("$task_name")
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "ERROR: No task directories found under $DATASET_ROOT"
    exit 1
fi

echo "======================================================================"
echo "  dataset       : $HF_DATASET_NAME"
echo "  tasks         : ${TASKS[*]}"
echo "  exp_name      : $EXP_NAME"
echo "  max_episodes  : $MAX_EPISODES"
echo "  gpu_id        : $GPU_ID"
echo "  server_port   : $SERVER_PORT"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Start policy server in background
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Starting policy server (log: $SERVER_LOG) ..."
bash "$EVAL_FILES_PATH/run_policy_server.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "      server PID: $SERVER_PID"

cleanup() {
    echo ""
    echo "Stopping policy server (PID $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait for server to be ready
# ---------------------------------------------------------------------------
echo "[2/3] Waiting for server on port $SERVER_PORT ..."
MAX_WAIT=120
elapsed=0
while ! nc -z 127.0.0.1 "$SERVER_PORT" 2>/dev/null; do
    if [[ $elapsed -ge $MAX_WAIT ]]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s. Check $SERVER_LOG"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "      Server ready after ${elapsed}s."

# ---------------------------------------------------------------------------
# Run eval for each task
# ---------------------------------------------------------------------------
echo "[3/3] Running evaluation ..."
echo ""

for task_name in "${TASKS[@]}"; do
    repo_id="$HF_DATASET_NAME/Randomized/$task_name"
    echo "----------------------------------------------------------------------"
    echo "  Task: $task_name  |  repo_id: $repo_id"
    echo "----------------------------------------------------------------------"

    HF_LEROBOT_HOME=$HF_LEROBOT_HOME \
    MODE=dataset \
    conda run -n "$ROBOTWIN_CONDA_ENV" --no-capture-output \
        bash "$EVAL_FILES_PATH/eval.sh" \
            "$repo_id" \
            "$task_name" \
            "$EXP_NAME" \
            "$MAX_EPISODES" \
            "$GPU_ID"

    echo ""
done

echo "======================================================================"
echo "All tasks done."
echo "======================================================================"
