#!/bin/bash
# eval.sh — two evaluation modes for starVLA on RoboTwin2
#
# MODE 1 — random (default): random scene rollouts via RoboTwin eval_policy.py
#   bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]
#
# MODE 2 — dataset: replay scenes from a LeRobot dataset (fixed seed + qpos)
#   MODE=dataset bash eval.sh <repo_id> <task_name> <exp_name> [max_episodes] [gpu_id]

ROBOTWIN_PATH=/shared/home/ZWA0839/Projects/VisualContextVLA/packages/RoboTwin

EVAL_FILES_PATH=$(cd "$(dirname "$0")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..
ROBOTWIN_DATA_PATH=$STARVLA_PATH/../../data/robotwin2

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

# MODE env var selects evaluation mode (default: random)
mode=${MODE:-random}

# ---------------------------------------------------------------------------
# MODE: random
# ---------------------------------------------------------------------------
if [[ "$mode" == "random" ]]; then
    policy_name="model2robotwin_interface"
    task_name=${1}
    task_config=${2}
    ckpt_setting=${3:-starvla_demo}
    seed=${4:-0}
    gpu_id=${5:-0}

    export CUDA_VISIBLE_DEVICES=${gpu_id}
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

    DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

    cd $ROBOTWIN_PATH

    echo "PYTHONPATH: $PYTHONPATH"

    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config $DEPLOY_POLICY_PATH \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --ckpt_setting ${ckpt_setting} \
        --seed ${seed} \
        --policy_name ${policy_name}

# ---------------------------------------------------------------------------
# MODE: dataset
# ---------------------------------------------------------------------------
elif [[ "$mode" == "dataset" ]]; then
    repo_id=${1}
    task_name=${2}
    exp_name=${3:-starvla_dataset_eval}
    max_episodes=${4:-none}
    gpu_id=${5:-0}

    export CUDA_VISIBLE_DEVICES=${gpu_id}
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
    echo -e "\033[33mDataset-mode eval: repo_id=${repo_id}, task=${task_name}, exp=${exp_name}\033[0m"

    cd $ROBOTWIN_DATA_PATH

    PYTHONWARNINGS=ignore::UserWarnin
    python $EVAL_FILES_PATH/eval_client.py \
        --repo-id "${repo_id}" \
        --task-name "${task_name}" \
        --exp-name "${exp_name}" \
        --max-episodes "${max_episodes}" \
        --policy-config "$EVAL_FILES_PATH/deploy_policy.yml"

else
    echo "Unknown MODE: $mode"
    echo "Usage:"
    echo "  bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]"
    echo "  MODE=dataset bash eval.sh <repo_id> <task_name> [exp_name] [max_episodes] [gpu_id]"
    exit 1
fi
