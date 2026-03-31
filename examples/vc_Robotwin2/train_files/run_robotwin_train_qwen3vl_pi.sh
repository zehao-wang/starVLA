# ---------------------------------------------------------------------------
# MACHINE config  — select with: MACHINE=a100 bash run_robotwin_train_qwen3vl_pi.sh
# ---------------------------------------------------------------------------
machine=${MACHINE:-l40s}

if [[ "$machine" == "l40s" ]]; then
    # L40S x 4
    export NCCL_SOCKET_IFNAME=enp39s0
    num_processes=4
    per_device_batch_size=1
elif [[ "$machine" == "a100" ]]; then
    # A100(40G) x 8
    export NCCL_SOCKET_IFNAME=ens32
    export NCCL_IB_HCA=ens65,ens129,ens161
    num_processes=8
    per_device_batch_size=1
else
    echo "Unknown MACHINE: $machine  (choices: l40s, a100)"
    exit 1
fi

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
MODE=${MODE:-train}   # usage: MODE=debug bash run_robotwin_train_qwen3vl_pi.sh

Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/vc_Robotwin2/train_files/starvla_cotrain_robotwin_qwen3vl_pi.yaml
run_root_dir=./results/Checkpoints

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
    data_mix=robotwin_all
    data_root=/shared/home/ZWA0839/Projects/VisualContextVLA/data/robotwin2/hf_lerobot/lerobot_robotwin_mixed_c40r10_vc_train
    per_device_batch_size=${per_device_batch_size}
    max_train_steps=100000
    num_warmup_steps=5000
    save_interval=10000
    logging_frequency=100
    eval_interval=5000
fi

run_id=0330_${data_mix}_qwen3pi_c40r10
# === End of environment variable configuration ===
###########################################################################################

echo "MODE: ${MODE} | data_mix: ${data_mix} | batch: ${per_device_batch_size} | steps: ${max_train_steps}"

# QwenPI uses a flow-matching diffusion head — no fast tokens needed, use base VLM directly.
# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin \
  --wandb_entity zekewang-ku-leuven



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
