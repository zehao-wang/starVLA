#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/shared/home/ZWA0839/.conda/envs/starVLA/bin/python
your_ckpt=results/Checkpoints/0129_robotwin_debug_qwen3fast_lora_c40r10/final_model/pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 2>&1 | tee examples/vc_Robotwin2/eval_files/out.log

# #################################
