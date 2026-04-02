#!/bin/bash
# Add FAST action special tokens to a Qwen VL model.
# Usage: bash examples/vc_Robotwin2_qwen35/train_files/add_fast_tokens.sh <source_model_dir> <target_model_dir>
# Example: bash add_fast_tokens.sh playground/Pretrained_models/Qwen3.5-2B \
#                                  playground/Pretrained_models/Qwen3.5-2B-Action

SOURCE_MODEL=${1}
TARGET_MODEL=${2}
FAST_TOKENS=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt
ADD_TOKENS_SCRIPT=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen35.py

if [ -z "$SOURCE_MODEL" ] || [ -z "$TARGET_MODEL" ]; then
    echo "[add_fast_tokens] ERROR: Usage: bash add_fast_tokens.sh <source_model_dir> <target_model_dir>"
    exit 1
fi

if [ -d "$TARGET_MODEL" ]; then
    echo "[add_fast_tokens] Target model already exists: ${TARGET_MODEL} — skipping."
else
    echo "[add_fast_tokens] Adding FAST action tokens: ${SOURCE_MODEL} -> ${TARGET_MODEL}"
    python ${ADD_TOKENS_SCRIPT} \
        --model-id ${SOURCE_MODEL} \
        --tokens-file ${FAST_TOKENS} \
        --save-dir ${TARGET_MODEL} \
        --init-strategy normal
    if [ $? -ne 0 ]; then
        echo "[add_fast_tokens] ERROR: Failed to add special tokens."
        exit 1
    fi
    echo "[add_fast_tokens] Done: ${TARGET_MODEL}"
fi
