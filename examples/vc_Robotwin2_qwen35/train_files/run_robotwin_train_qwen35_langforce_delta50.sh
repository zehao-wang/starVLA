#!/bin/bash
# ---------------------------------------------------------------------------
# Single-node multi-GPU training (default).
# Multi-node: called via srun from the .sbatch file, which exports
#   GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, SLURM_NNODES, SLURM_PROCID
#   and cluster-specific NCCL_* vars before invoking this script.
#
# Usage (single-node):
#   MACHINE=l40s bash run_robotwin_train_qwen35_langforce_delta50.sh
#   MODE=debug   bash run_robotwin_train_qwen35_langforce_delta50.sh
# ---------------------------------------------------------------------------

set -e

# Resolve repository root from this script location, independent of caller cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# MACHINE config — single-node defaults; sbatch overrides via env vars
# ---------------------------------------------------------------------------
machine=${MACHINE:-l40s}

if [[ "$machine" == "l40s" ]]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp39s0}
    per_device_batch_size=1
elif [[ "$machine" == "a100" ]]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens32}
    export NCCL_IB_HCA=${NCCL_IB_HCA:-ens65,ens129,ens161}
    per_device_batch_size=4
elif [[ "$machine" == "h100" ]]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp}
    export FI_EFA_USE_DEVICE_RDMA=${FI_EFA_USE_DEVICE_RDMA:-1}
    export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib64:${LD_LIBRARY_PATH}
    per_device_batch_size=8
elif [[ "$machine" == "h200" ]]; then
    # p5en.48xlarge — EFA (no InfiniBand), 16x enp* NICs, 8x H200 143GB
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp}
    export FI_EFA_USE_DEVICE_RDMA=${FI_EFA_USE_DEVICE_RDMA:-1}
    export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib64:${LD_LIBRARY_PATH}
    per_device_batch_size=32
else
    echo "Unknown MACHINE: $machine  (choices: l40s, a100, h200)"
    exit 1
fi

# NCCL — allow sbatch to override via exported env vars
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
MODE=${MODE:-train}

Framework_name=LangForce
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-2B-Action-Query
config_yaml=./examples/vc_Robotwin2_qwen35/train_files/starvla_cotrain_robotwin_qwen35_langforce.yaml
run_root_dir=./results/Checkpoints
action_mode=delta
normalization_mode=q99

if [ "$MODE" = "debug" ]; then
    data_root=/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot/lerobot_robotwin_rand20k_debug
    data_mix=robotwin_debug
    per_device_batch_size=1
    max_train_steps=50
    num_warmup_steps=20
    save_interval=10
    logging_frequency=10
    eval_interval=25
else
    data_mix=robotwin_all_50
    data_root=/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot/lerobot_robotwin_mixed_c40r450_vc_train
    per_device_batch_size=${per_device_batch_size}
    max_train_steps=50000
    num_warmup_steps=5000
    save_interval=1000
    logging_frequency=100
    eval_interval=5000
fi

run_id=260409_${machine}_${data_mix}_qwen35_langforce_delta50_c40r10

echo "MODE: ${MODE} | data_mix: ${data_mix} | batch: ${per_device_batch_size} | steps: ${max_train_steps}"

# ---------------------------------------------------------------------------
# Output dir & auto-resume
# ---------------------------------------------------------------------------
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# ---------------------------------------------------------------------------
# Auto-generate eval artifacts in output_dir
#   via shared helper script
# ---------------------------------------------------------------------------
bash "${REPO_ROOT}/examples/vc_Robotwin2_qwen35/eval_files/batch_submission/generate_eval_artifacts.sh" \
    "${REPO_ROOT}" \
    "${output_dir}" \
    "${run_id}" \
    "${action_mode}" \
    "${normalization_mode}"

is_resume_flag=""
if compgen -G "${output_dir}/checkpoints/steps_*" > /dev/null 2>&1; then
    echo "Checkpoint found in ${output_dir}/checkpoints — resuming."
    is_resume_flag="--trainer.is_resume True"
else
    echo "No checkpoint found — training from scratch."
fi

# ---------------------------------------------------------------------------
# Distributed args — single-node vs multi-node (driven by SLURM env vars)
# SLURM sets CUDA_VISIBLE_DEVICES to allocated GPUs; torch.cuda.device_count()
# therefore always reflects the correct per-node GPU count.
# ---------------------------------------------------------------------------
gpus_per_node=$(python3 -c "import torch; print(torch.cuda.device_count())")
num_nodes=${SLURM_NNODES:-1}

if [[ "$num_nodes" -gt 1 ]]; then
    echo "Multi-node: nodes=${num_nodes}  gpus/node=${gpus_per_node}  rank=${SLURM_PROCID}"
    dist_args="
        --main_process_ip ${MASTER_ADDR}
        --main_process_port ${MASTER_PORT}
        --machine_rank ${SLURM_PROCID}
        --num_machines ${num_nodes}
        --num_processes $((gpus_per_node * num_nodes))
    "
else
    echo "Single-node: gpus=${gpus_per_node}"
    dist_args="--num_processes ${gpus_per_node}"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# Point triton autotune cache to a job-scoped tmp dir.
# Empty string causes DeepSpeed to set file_path=None → TypeError; must be a valid path.
# The atexit write-race between GPU processes is harmless noise.
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}
mkdir -p "${TRITON_CACHE_DIR}"

# export WANDB_MODE=disabled
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  ${dist_args} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.future_action_window_size 49 \
  --framework.action_model.action_horizon 50 \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.action_mode ${action_mode} \
  --datasets.vla_data.action_type delta_qpos \
  --datasets.vla_data.normalization_mode ${normalization_mode} \
  --datasets.vla_data.action_mode_apply_keys "[action.left_joints,action.right_joints]" \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin \
  --wandb_entity zekewang-ku-leuven \
  ${is_resume_flag} \
  2>&1 | tee -a "${output_dir}/train.log"
