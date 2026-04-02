#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_API_KEY_FILE="/shared/home/ZWA0839/Projects/VisualContextVLA/packages/starVLA/examples/vc_Robotwin2_qwen35/train_files/WANDB_API_KEY"
OFFLINE_RUN="${SCRIPT_DIR}/wandb/wandb/offline-run-20260331_225519-260331_robotwin_all_qwen35_pi_lora_c40r10"

INTERVAL=${1:-1800}  # 默认30分钟，可通过第一个参数指定秒数

echo "Starting periodic wandb sync every ${INTERVAL}s..."
echo "Run URL: https://wandb.ai/zekewang-ku-leuven/starVLA_Robotwin/runs/260331_robotwin_qwen35_pi_lora_v2"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing..."
    WANDB_API_KEY=$(cat "${WANDB_API_KEY_FILE}") wandb sync \
        "${OFFLINE_RUN}" \
        --entity zekewang-ku-leuven \
        --project starVLA_Robotwin \
        --id 260331_robotwin_qwen35_pi_lora_v2 \
        --no-mark-synced
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Next sync in ${INTERVAL}s."
    sleep "${INTERVAL}"
done
