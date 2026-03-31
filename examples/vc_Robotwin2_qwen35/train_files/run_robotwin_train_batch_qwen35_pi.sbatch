#!/bin/bash
#SBATCH --job-name=starvla_qwen35pi
#SBATCH -p spot-p5-8gpu-h100-h200
#SBATCH --constraint=spot-p5en-48xlarge
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=results/Checkpoints/slurm-%x-%j.out
#SBATCH --error=results/Checkpoints/slurm-%x-%j.err

set -e

# ---------------------------------------------------------------------------
# Cluster network config — p5en.48xlarge (EFA, aws-ofi-nccl, no InfiniBand)
# ---------------------------------------------------------------------------
# Load aws-ofi-nccl plugin (EFA transport)
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib64:${LD_LIBRARY_PATH}

# Use EFA network interfaces (enp71s0 ~ enp149s0, 16 NICs)
export NCCL_SOCKET_IFNAME=enp
# Enable GPU Direct RDMA over EFA
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# ---------------------------------------------------------------------------
# Distributed env (consumed by run_robotwin_train_qwen35_pi.sh)
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

srun MACHINE=h200 MODE=train bash "${SCRIPT_DIR}/run_robotwin_train_qwen35_pi.sh"
