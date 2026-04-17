#!/usr/bin/env python
"""Small SimplerEnv/SAPIEN diagnostics for vc_simplerenv_qwen35.

Run this with the simpler_env Python, for example:

PYTHONPATH="$PWD:/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv" \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
/shared/home/ZWA0839/.conda/envs/simpler_env/bin/python \
  examples/vc_simplerenv_qwen35/eval_files/test_simplerenv_setup.py

The script intentionally checks one layer at a time so install/driver issues
are easier to separate from model-server issues.
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def check_import(module_name: str) -> object | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        print(f"[FAIL] import {module_name}")
        traceback.print_exc()
        return None

    version = getattr(module, "__version__", None)
    location = getattr(module, "__file__", None)
    print(f"[ OK ] import {module_name}")
    if version is not None:
        print(f"       version : {version}")
    if location is not None:
        print(f"       file    : {location}")
    return module


def check_path(label: str, path: str | Path) -> bool:
    p = Path(path)
    ok = p.exists()
    print(f"[{' OK ' if ok else 'FAIL'}] {label}: {p}")
    return ok


def run_probe(label: str, command: list[str]) -> None:
    exe = shutil.which(command[0])
    print(f"{label:<24}: {exe or 'not found'}")
    if exe is None:
        return
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception:
        print(f"{label} probe failed:")
        traceback.print_exc()
        return
    lines = result.stdout.strip().splitlines()
    print(f"{label} returncode       : {result.returncode}")
    for line in lines[:12]:
        print(f"  {line}")
    if len(lines) > 12:
        print(f"  ... ({len(lines) - 12} more lines)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simplerenv-path",
        default="/shared/home/ZWA0839/Projects/VisualContextVLA/packages/SimplerEnv",
        help="Path to the SimplerEnv repository.",
    )
    parser.add_argument(
        "--env-name",
        default="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        help="SimplerEnv/ManiSkill2 env id to build.",
    )
    parser.add_argument("--scene-name", default="bridge_table_1_v1")
    parser.add_argument("--robot", default="widowx")
    parser.add_argument(
        "--overlay",
        default=None,
        help="RGB overlay path. Defaults to bridge_real_eval_1.png under SimplerEnv.",
    )
    parser.add_argument(
        "--skip-env-build",
        action="store_true",
        help="Only test imports and SapienRenderer; do not build ManiSkill2 env.",
    )
    parser.add_argument(
        "--step-smoke",
        action="store_true",
        help="After env build/reset, run one zero-action env.step smoke test.",
    )
    parser.add_argument(
        "--offscreen-only",
        action="store_true",
        help="Pass offscreen_only=True to SapienRenderer and env renderer_kwargs.",
    )
    parser.add_argument(
        "--renderer-device",
        default=None,
        help="Optional SapienRenderer device, e.g. cuda:0.",
    )
    parser.add_argument(
        "--shader-dir",
        default=None,
        help="Optional shader_dir override, e.g. trivial or ibl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simpler = Path(args.simplerenv_path)
    overlay = Path(args.overlay) if args.overlay else (
        simpler / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
    )

    banner("1. Process Environment")
    print(f"python                  : {sys.executable}")
    print(f"cwd                     : {Path.cwd()}")
    print(f"PYTHONPATH              : {os.environ.get('PYTHONPATH')}")
    print(f"CUDA_VISIBLE_DEVICES    : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"VK_ICD_FILENAMES        : {os.environ.get('VK_ICD_FILENAMES')}")
    print(f"LD_LIBRARY_PATH         : {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"DISPLAY                 : {os.environ.get('DISPLAY')}")
    print(f"NVIDIA_VISIBLE_DEVICES  : {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    print(f"NVIDIA_DRIVER_CAPS      : {os.environ.get('NVIDIA_DRIVER_CAPABILITIES')}")
    print(f"ctypes libvulkan        : {ctypes.util.find_library('vulkan')}")

    banner("2. Path Checks")
    paths_ok = True
    paths_ok &= check_path("SimplerEnv repo", simpler)
    paths_ok &= check_path("ManiSkill2_real2sim", simpler / "ManiSkill2_real2sim")
    paths_ok &= check_path("overlay image", overlay)
    if os.environ.get("VK_ICD_FILENAMES"):
        for idx, icd in enumerate(os.environ["VK_ICD_FILENAMES"].split(":")):
            paths_ok &= check_path(f"VK ICD {idx}", icd)

    banner("2b. Vulkan/NVIDIA Probes")
    for icd_dir in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d"):
        p = Path(icd_dir)
        if p.exists():
            print(f"{icd_dir}:")
            for child in sorted(p.glob("*.json")):
                print(f"  {child}")
        else:
            print(f"{icd_dir}: missing")
    run_probe("nvidia-smi", ["nvidia-smi", "-L"])
    run_probe("vulkaninfo", ["vulkaninfo", "--summary"])

    banner("3. Import Checks")
    numpy = check_import("numpy")
    gymnasium = check_import("gymnasium")
    sapien_pkg = check_import("sapien")
    sapien_core = check_import("sapien.core")
    check_import("mani_skill2_real2sim")
    check_import("simpler_env")
    check_import("simpler_env.utils.env.env_builder")
    if None in (numpy, gymnasium, sapien_pkg, sapien_core):
        return 2

    banner("4. SAPIEN Renderer Smoke Test")
    renderer_kwargs = {}
    if args.offscreen_only:
        renderer_kwargs["offscreen_only"] = True
    if args.renderer_device:
        renderer_kwargs["device"] = args.renderer_device
    print(f"renderer_kwargs         : {renderer_kwargs}")
    try:
        renderer = sapien_core.SapienRenderer(**renderer_kwargs)
        print("[ OK ] SapienRenderer created")
        del renderer
    except Exception:
        print("[FAIL] SapienRenderer creation failed")
        traceback.print_exc()
        return 3

    if args.skip_env_build:
        banner("Done")
        print("Skipped env build by request.")
        return 0 if paths_ok else 1

    banner("5. ManiSkill2 Env Build Smoke Test")
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    build_kwargs = {
        "obs_mode": "rgbd",
        "robot": args.robot,
        "sim_freq": 500,
        "control_freq": 5,
        "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "scene_name": args.scene_name,
        "camera_cfgs": {"add_segmentation": True},
        "rgb_overlay_path": str(overlay),
        "max_episode_steps": 2,
    }
    if renderer_kwargs:
        build_kwargs["renderer_kwargs"] = renderer_kwargs
    if args.shader_dir:
        build_kwargs["shader_dir"] = args.shader_dir

    print(f"env_name                : {args.env_name}")
    print(f"build_kwargs            : {build_kwargs}")
    try:
        env = build_maniskill2_env(args.env_name, **build_kwargs)
        print(f"[ OK ] env built: {type(env).__name__}")
        if args.step_smoke:
            reset_options = {
                "robot_init_options": {
                    "init_xy": np.array([0.147, 0.028], dtype=np.float32),
                    "init_rot_quat": np.array([0, 0, 0, 1], dtype=np.float32),
                },
                "obj_init_options": {"episode_id": 0},
            }
            print(f"reset_options           : {reset_options}")
            obs, _ = env.reset(options=reset_options)
            print(f"[ OK ] env reset; obs type: {type(obs).__name__}")
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            if action.shape[0] >= 1:
                action[-1] = 1.0
            print(f"step_action             : shape={action.shape}, dtype={action.dtype}, value={action}")
            obs, reward, done, truncated, info = env.step(np.ascontiguousarray(action))
            print(f"[ OK ] env step; reward={reward}, done={done}, truncated={truncated}, info={info}")
        env.close()
        print("[ OK ] env closed")
    except Exception:
        print("[FAIL] env build failed")
        traceback.print_exc()
        return 4

    banner("Done")
    print("All Python-level checks passed. If your eval still fails, inspect policy/server logs next.")
    return 0 if paths_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
