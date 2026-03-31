#!/bin/bash
# run_policy_server_gpu.sh — Start ONE policy server on a specific GPU + port.
#
# Usage:
#   bash run_policy_server_gpu.sh <gpu_id> <port> <ckpt_path> [extra args …]
#
# Called by run_eval_distributed.sh for each GPU.
# Must be run from packages/starVLA/ with the starVLA conda env active.

set -e

GPU_ID=${1:?Usage: run_policy_server_gpu.sh <gpu_id> <port> <ckpt_path>}
PORT=${2:?Usage: run_policy_server_gpu.sh <gpu_id> <port> <ckpt_path>}
CKPT_PATH=${3:?Usage: run_policy_server_gpu.sh <gpu_id> <port> <ckpt_path>}
shift 3
EXTRA_ARGS="$*"

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/shared/home/ZWA0839/.conda/envs/starVLA/bin/python}

echo "[Server gpu=${GPU_ID} port=${PORT}] Starting …"
echo "  ckpt : ${CKPT_PATH}"
echo "  extra: ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${STAR_VLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        ${EXTRA_ARGS}
