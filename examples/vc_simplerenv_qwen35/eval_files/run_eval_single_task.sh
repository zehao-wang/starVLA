#!/bin/bash
# Single-task SimplerEnv eval — starts policy server, runs one task, exits.
# Each sbatch job runs this script for one task on one GPU.
#
# Required env vars:
#   TASK_NAME   — SimplerEnv environment ID
#   CKPT_PATH   — path to pytorch_model.pt
#
# Optional overrides:
#   PORT, OBJ_EPISODE_{START,END}, PREFLIGHT_SIM
#   STAR_VLA_PYTHON, SIM_PYTHON, SIMPLERENV_PATH, REPO_ROOT
#   SIM_RENDERER_DEVICE, SIM_VK_ICD, SIM_KEEP_LD_LIBRARY_PATH, MAX_WAIT
#
# Manual usage from packages/starVLA:
#   TASK_NAME=PutCarrotOnPlateInScene-v0 \
#   CKPT_PATH=results/Checkpoints/.../final_model/pytorch_model.pt \
#   bash examples/vc_simplerenv_qwen35/eval_files/run_eval_single_task.sh

set -eo pipefail

EVAL_FILES_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(realpath "$EVAL_FILES_DIR/../../..")}

# ---- Required ---------------------------------------------------------------
CKPT_PATH=${CKPT_PATH:-}
TASK_NAME=${TASK_NAME:-}

# ---- Optional overrides -----------------------------------------------------
PORT=${PORT:-6780}
OBJ_EPISODE_START=${OBJ_EPISODE_START:-0}
OBJ_EPISODE_END=${OBJ_EPISODE_END:-24}
PREFLIGHT_SIM=${PREFLIGHT_SIM:-0}
# EXP_NAME: auto-derived from checkpoint path (.../Checkpoints/<run_id>/final_model/ckpt.pt)
# Override explicitly if evaluating intermediate checkpoints.
EXP_NAME=${EXP_NAME:-}
DATASET_NAME=${DATASET_NAME:-bridge_rt_1}

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}
SIM_PYTHON=${SIM_PYTHON:-/shared/home/ZWA0839/.conda/envs/simpler_env/bin/python}
SIMPLERENV_PATH=${SIMPLERENV_PATH:-/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv}

SIM_RENDERER_DEVICE=${SIM_RENDERER_DEVICE:-cuda:0}
SIM_VK_ICD=${SIM_VK_ICD:-}
if [[ -z "$SIM_VK_ICD" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    SIM_VK_ICD=/etc/vulkan/icd.d/nvidia_icd.json
fi
SIM_KEEP_LD_LIBRARY_PATH=${SIM_KEEP_LD_LIBRARY_PATH:-1}
MAX_WAIT=${MAX_WAIT:-360}

# ---- Validate ---------------------------------------------------------------
if [[ -z "$TASK_NAME" ]]; then
    echo "ERROR: TASK_NAME is empty."
    exit 1
fi
if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: CKPT_PATH is empty."
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi
if [[ ! -x "$STAR_VLA_PYTHON" ]]; then
    echo "ERROR: STAR_VLA_PYTHON not executable: $STAR_VLA_PYTHON"
    exit 1
fi
if [[ ! -x "$SIM_PYTHON" ]]; then
    echo "ERROR: SIM_PYTHON not executable: $SIM_PYTHON"
    exit 1
fi
if [[ ! -d "$SIMPLERENV_PATH" ]]; then
    echo "ERROR: SIMPLERENV_PATH not found: $SIMPLERENV_PATH"
    exit 1
fi

# ---- Task → scene config ----------------------------------------------------
case "$TASK_NAME" in
    StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
    |PutCarrotOnPlateInScene-v0 \
    |PutSpoonOnTableClothInScene-v0)
        scene_name=bridge_table_1_v1
        robot=widowx
        overlay="$SIMPLERENV_PATH/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
        robot_init_x=0.147
        robot_init_y=0.028
        ;;
    PutEggplantInBasketScene-v0)
        scene_name=bridge_table_1_v2
        robot=widowx_sink_camera_setup
        overlay="$SIMPLERENV_PATH/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
        robot_init_x=0.127
        robot_init_y=0.06
        ;;
    *)
        echo "ERROR: Unknown TASK_NAME: $TASK_NAME"
        echo "Supported: StackGreenCubeOnYellowCubeBakedTexInScene-v0  PutCarrotOnPlateInScene-v0"
        echo "           PutSpoonOnTableClothInScene-v0  PutEggplantInBasketScene-v0"
        exit 1
        ;;
esac

# ---- Paths & dirs -----------------------------------------------------------
CKPT_NAME=$(basename "$CKPT_PATH")
CKPT_NAME=${CKPT_NAME%.*}

# Auto-derive EXP_NAME from path: .../Checkpoints/<run_id>/final_model/ckpt.pt
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME=$(basename "$(dirname "$(dirname "$(realpath "$CKPT_PATH")")")")
fi

# Centralised results root, mirroring results/robotwin2_dataset/
EVAL_RESULTS_ROOT="${REPO_ROOT}/results/simplerenv_dataset"
TASK_OUTPUT_DIR="${EVAL_RESULTS_ROOT}/${EXP_NAME}/${DATASET_NAME}_${TASK_NAME}"
SERVER_LOG_DIR="${EVAL_RESULTS_ROOT}/${EXP_NAME}/logs"
mkdir -p "$TASK_OUTPUT_DIR" "$SERVER_LOG_DIR"

SERVER_LOG="${SERVER_LOG_DIR}/policy_server_${PORT}.log"
TASK_LOG="${TASK_OUTPUT_DIR}/${CKPT_NAME}.log"

export PYTHONPATH="$REPO_ROOT:$SIMPLERENV_PATH:${PYTHONPATH:-}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$REPO_ROOT"

echo "======================================================================"
echo "  task           : $TASK_NAME"
echo "  ckpt           : $CKPT_PATH"
echo "  port           : $PORT"
echo "  sim_renderer   : $SIM_RENDERER_DEVICE"
echo "  sim_vk_icd     : ${SIM_VK_ICD:-unset}"
echo "  preflight_sim  : $PREFLIGHT_SIM"
echo "  obj episodes   : [$OBJ_EPISODE_START, $OBJ_EPISODE_END)"
echo "  star_vla_python: $STAR_VLA_PYTHON"
echo "  sim_python     : $SIM_PYTHON"
echo "  simplerenv     : $SIMPLERENV_PATH"
echo "  exp_name       : $EXP_NAME"
echo "  dataset_name   : $DATASET_NAME"
echo "  task_output    : $TASK_OUTPUT_DIR"
echo "  server_log     : $SERVER_LOG"
echo "  task_log       : $TASK_LOG"
echo "======================================================================"
echo ""

build_sim_env_prefix() {
    sim_env=(env)
    sim_env_vars=()
    if [[ -n "$SIM_VK_ICD" ]]; then
        sim_env_vars+=(VK_ICD_FILENAMES="$SIM_VK_ICD")
    else
        sim_env+=(-u VK_ICD_FILENAMES)
    fi
    if [[ "$SIM_KEEP_LD_LIBRARY_PATH" != "1" ]]; then
        sim_ld_path=$(
            "$SIM_PYTHON" - <<'PY'
import os
parts = [
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if p and "/tmp/nvidia_extract" not in p
]
print(":".join(parts))
PY
        )
        sim_env_vars+=(LD_LIBRARY_PATH="$sim_ld_path")
    fi
    sim_env+=("${sim_env_vars[@]}")
}

# ---- Preflight --------------------------------------------------------------
if [[ "$PREFLIGHT_SIM" == "1" ]]; then
    echo "[0/3] Preflight SimplerEnv build ..."
    build_sim_env_prefix
    "${sim_env[@]}" "$SIM_PYTHON" "$EVAL_FILES_DIR/test_simplerenv_setup.py" \
        --simplerenv-path "$SIMPLERENV_PATH" \
        --env-name "$TASK_NAME" \
        --scene-name "$scene_name" \
        --robot "$robot"
    echo ""
fi

# ---- Policy server ----------------------------------------------------------
echo "[1/3] Starting policy server ..."
"$STAR_VLA_PYTHON" -u deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT_PATH" \
    --port "$PORT" \
    --use_bf16 \
    --idle_timeout -1 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    PID $SERVER_PID, log: $SERVER_LOG"

cleanup() {
    echo ""
    echo "Stopping policy server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---- Wait for server --------------------------------------------------------
echo ""
echo "[2/3] Waiting for server ..."
port_open() {
    "$STAR_VLA_PYTHON" -c "
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
while ! port_open "$PORT"; do
    if [[ $elapsed -ge $MAX_WAIT ]]; then
        echo "ERROR: server did not start within ${MAX_WAIT}s. Check $SERVER_LOG"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "    port $PORT ready after ${elapsed}s"

# ---- Run task ---------------------------------------------------------------
echo ""
echo "[3/3] Running task: $TASK_NAME ..."

sim_cmd=(
    "$SIM_PYTHON" "$EVAL_FILES_DIR/start_simpler_env.py"
    --ckpt-path "$CKPT_PATH"
    --host 127.0.0.1
    --port "$PORT"
    --robot "$robot"
    --policy-setup widowx_bridge
    --control-freq 5
    --sim-freq 500
    --max-episode-steps 120
    --env-name "$TASK_NAME"
    --scene-name "$scene_name"
    --rgb-overlay-path "$overlay"
    --robot-init-x-range "$robot_init_x" "$robot_init_x" 1
    --robot-init-y-range "$robot_init_y" "$robot_init_y" 1
    --obj-variation-mode episode
    --obj-episode-range "$OBJ_EPISODE_START" "$OBJ_EPISODE_END"
    --robot-init-rot-quat-center 0 0 0 1
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
    --logging-dir "$TASK_OUTPUT_DIR"
)
if [[ -n "$SIM_RENDERER_DEVICE" ]]; then
    sim_cmd+=(--renderer-device "$SIM_RENDERER_DEVICE")
fi

build_sim_env_prefix
"${sim_env[@]}" "${sim_cmd[@]}" 2>&1 | tee "$TASK_LOG"

echo ""
echo "======================================================================"
echo "Task complete. Log: $TASK_LOG"
echo "======================================================================"
