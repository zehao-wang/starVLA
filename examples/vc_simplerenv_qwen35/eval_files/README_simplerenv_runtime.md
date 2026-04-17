# SimplerEnv Runtime Notes

This directory contains the local evaluation wrapper for running a starVLA
policy server against SimplerEnv/ManiSkill2_real2sim.

The setup intentionally uses two Python environments:

- `starVLA`: loads the checkpoint and runs `deployment/model_server/server_policy.py`.
- `simpler_env`: runs SAPIEN, ManiSkill2_real2sim, and the SimplerEnv rollout.

Keep SimplerEnv-specific patches in `examples/vc_simplerenv_qwen35/eval_files`.
Treat `examples/SimplerEnv` and the external `packages/SimplerEnv` checkout as
reference code unless a change is explicitly needed there.

## Required Runtime Patch

On the A100 SimplerEnv setup tested here, `sapien==2.2.2` and
ManiSkill2_real2sim were unstable with `numpy==2.2.6`: the evaluation could
build and reset the environment, then segfault on the first `env.step()` inside
`pd_ee_pose.py::compute_fk`.

The stable simulator-side versions are:

```text
numpy==1.24.0
opencv-python==4.6.0.66
sapien==2.2.2
gymnasium==0.29.1
```

Apply the local patch requirements inside the simulator env:

```bash
conda activate simpler_env
python -m pip install -r examples/vc_simplerenv_qwen35/eval_files/requirements_simplerenv_patch.txt
```

Do not apply this patch to the `starVLA` policy-server env unless you are
intentionally rebuilding that environment too.

`start_simpler_env.py` checks these simulator-side versions at startup. If the
versions do not match, it exits and points back to this file instead of trying
to recover with runtime workarounds.

## Vulkan ICD

This node exposes both NVIDIA and Mesa Vulkan ICD files. Letting the Vulkan
loader auto-enumerate all ICDs caused SAPIEN renderer creation to fail with:

```text
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
```

`run_eval_local.sh` therefore sets the simulator subprocess to use the NVIDIA
ICD when available:

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

`run_eval_local.sh` sets the simulator subprocess with
`CUDA_VISIBLE_DEVICES=$SIM_GPU_ID` and keeps the SAPIEN renderer device at the
local index `--renderer-device cuda:0`.

Note: SAPIEN/Vulkan device selection can be confusing under
`CUDA_VISIBLE_DEVICES`. Depending on the driver/runtime combination, the
renderer may follow CUDA's visible-device remapping or may still appear under a
physical GPU index in `nvidia-smi`. If GPU placement looks wrong, first confirm
the banner values from `run_eval_local.sh`, then use
`test_simplerenv_setup.py --renderer-device cuda:0` inside the same environment
to smoke-test the mapping.

## Smoke Tests

Run the simulator-only checks before starting a policy server:

```bash
unset CUDA_VISIBLE_DEVICES SIM_GPU_ID SIM_VK_ICD
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

PYTHONPATH="$PWD:/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv" \
MPLCONFIGDIR=/tmp/matplotlib-$USER \
/shared/home/ZWA0839/.conda/envs/simpler_env/bin/python \
  examples/vc_simplerenv_qwen35/eval_files/test_simplerenv_setup.py \
  --step-smoke
```

Expected result:

- `SapienRenderer created`
- `env built`
- `env reset`
- `env step`
- no segmentation fault

Then run a one-episode end-to-end smoke eval:

```bash
OBJ_EPISODE_END=1 PREFLIGHT_SIM=1 \
bash results/Checkpoints/260412_a100_bridge_rt_1_qwen35_gr00t_simplerenv/run_eval_local.sh
```

## Single-Task Slurm Evaluation

For full evaluation, use one sbatch job per SimplerEnv task. Each job requests
one GPU and then lets both the starVLA policy server and SAPIEN simulator use
the default in-process `cuda:0`.

Submit all listed tasks:

```bash
CKPT_PATH=results/Checkpoints/260412_a100_bridge_rt_1_qwen35_gr00t_simplerenv/final_model/pytorch_model.pt \
bash examples/vc_simplerenv_qwen35/eval_files/batch_submission/submit_multi_tasks.sh
```

Run just one task locally or inside an interactive single-GPU allocation:

```bash
TASK_NAME=PutCarrotOnPlateInScene-v0 \
OBJ_EPISODE_END=1 \
bash examples/vc_simplerenv_qwen35/eval_files/run_eval_single_task.sh
```

The submit script is intentionally similar to the Robotwin2 eval workflow:
`submit_multi_tasks.sh` submits one `run_eval_single_task.sbatch` job per task,
and each job calls `run_eval_single_task.sh`.

## Debugging History

Observed failure sequence before the patch:

1. Without forcing the NVIDIA ICD, `SapienRenderer()` failed with
   `ErrorExtensionNotPresent`.
2. After forcing the NVIDIA ICD, renderer/env creation passed, but NumPy 2.x
   led to a native segfault during the first `env.step()`.
3. Downgrading to `numpy==1.24.0` and `opencv-python==4.6.0.66` allowed full
   one-episode rollouts to complete. Example logs reached `evaluator done`,
   with `StackGreenCubeOnYellowCubeBakedTexInScene-v0` returning
   `success_arr=[False]` and `PutCarrotOnPlateInScene-v0` returning
   `success_arr=[True]`.
