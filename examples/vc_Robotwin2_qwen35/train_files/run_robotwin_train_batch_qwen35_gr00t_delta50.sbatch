#!/bin/bash
#SBATCH --job-name=delta50_qwen35gr00t
#SBATCH -p ondemand-p4d-8gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=_slurm_logs/slurm-%x-%j.out
#SBATCH --error=_slurm_logs/slurm-%x-%j.err
#SBATCH --requeue

set -e

# ---------------------------------------------------------------------------
# Load aws-ofi-nccl plugin (EFA transport)
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib64:${LD_LIBRARY_PATH}

# Enable GPU Direct RDMA over EFA
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# ---------------------------------------------------------------------------
# Distributed env (consumed by run_robotwin_train_qwen35_gr00t_delta50.sh)
# GPUS_PER_NODE is intentionally omitted — the sh script auto-detects via
# torch.cuda.device_count() which respects SLURM's CUDA_VISIBLE_DEVICES.
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

echo "SLURM_NNODES=${SLURM_NNODES}  MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

# ---------------------------------------------------------------------------
# Run — all training config lives in the sh script
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/run_robotwin_train_qwen35_gr00t_delta50.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]] && [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/run_robotwin_train_qwen35_gr00t_delta50.sh" ]]; then
	TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/run_robotwin_train_qwen35_gr00t_delta50.sh"
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
	echo "Cannot find run script: run_robotwin_train_qwen35_gr00t_delta50.sh"
	echo "Checked: ${SCRIPT_DIR} and ${SLURM_SUBMIT_DIR:-<unset>}"
	exit 1
fi

srun --export=ALL,MACHINE=a100,MODE=train bash "${TRAIN_SCRIPT}"
