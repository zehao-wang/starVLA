"""
utils/robotwin_helpers.py — shared helpers for RoboTwin2 dataset-mode eval.

Extracted from eval_client.py so both the single-GPU client and the
distributed orchestrator/worker can import the same logic without duplication.

Requires PYTHONPATH to include:
  - packages/RoboTwin          (for envs.*, envs.utils.create_actor)
  - packages/starVLA           (for starVLA.*)
  - eval_files/                (for model2robotwin_interface)
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# RoboTwin imports (require PYTHONPATH)
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError


# ---------------------------------------------------------------------------
# LeRobot dataset helpers
# ---------------------------------------------------------------------------

def _get_lerobot_home() -> Path:
    return Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot"),
        )
    )


def _load_test_metadata(repo_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta_dir = _get_lerobot_home() / repo_id / "meta"
    init_states: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    with open(meta_dir / "env_init_states.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                init_states.append(json.loads(line))

    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        with open(episodes_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    else:
        parquet_files = sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))
        if parquet_files:
            try:
                import pandas as pd
                for parquet_path in parquet_files:
                    frame = pd.read_parquet(parquet_path, columns=["episode_index", "tasks", "length"])
                    for row in frame.to_dict(orient="records"):
                        episodes.append({
                            "episode_index": int(row["episode_index"]),
                            "tasks": row.get("tasks", []),
                            "length": int(row.get("length", 0)),
                        })
                episodes.sort(key=lambda item: int(item["episode_index"]))
            except (ImportError, Exception):
                pass

    if not episodes:
        for idx, s in enumerate(init_states):
            episodes.append({"episode_index": idx, "tasks": [s["task_name"]], "length": 0})

    if len(init_states) != len(episodes):
        raise ValueError(f"Mismatch: {len(init_states)} init_states vs {len(episodes)} episodes")
    return init_states, episodes


def _build_episode_records(
    init_states: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def _extract_instruction(tasks: Any) -> str:
        if tasks is None:
            return ""
        if isinstance(tasks, (list, tuple)):
            return str(tasks[0]) if tasks else ""
        if isinstance(tasks, np.ndarray):
            flat = tasks.reshape(-1)
            return str(flat[0]) if flat.size > 0 else ""
        return str(tasks)

    records = []
    for init_s, ep in zip(init_states, episodes):
        records.append({
            "episode_index": int(ep["episode_index"]),
            "instruction": _extract_instruction(ep.get("tasks", [])),
            "seed": int(init_s["seed"]),
            "initial_qpos": np.asarray(init_s["initial_qpos"], dtype=np.float32),
            "task_name": str(init_s["task_name"]),
            "task_config": str(init_s["task_config"]),
            "length": int(ep.get("length", 0)),
        })
    return records


# ---------------------------------------------------------------------------
# RoboTwin environment helpers
# ---------------------------------------------------------------------------

def build_env_args(task_name: str, task_config: str, policy_name: str) -> dict[str, Any]:
    configs = Path(CONFIGS_PATH)
    with open(configs / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["policy_name"] = policy_name

    with open(configs / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    with open(configs / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = yaml.safe_load(f)

    def _get_embodiment_file(t: str) -> str:
        rf = embodiment_types[t]["file_path"]
        if rf is None:
            raise RuntimeError(f"No embodiment file for type: {t}")
        return rf

    def _get_embodiment_config(robot_file: str) -> dict:
        with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    embodiment_type = args.get("embodiment")
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment list must have 1 or 3 entries")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    return args


def _load_step_limit(task_name: str) -> int:
    with open(Path(CONFIGS_PATH) / "_eval_step_limit.yml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if task_name in data:
        return int(data[task_name])
    raise KeyError(f"Step limit for task '{task_name}' not found in _eval_step_limit.yml")


def _set_robot_joints(task_env, qpos: np.ndarray, settle_steps: int = 100) -> None:
    qpos = np.asarray(qpos, dtype=np.float32)
    for i, joint in enumerate(task_env.robot.left_arm_joints):
        joint.set_drive_target(float(qpos[i]))
    for i, joint in enumerate(task_env.robot.right_arm_joints):
        joint.set_drive_target(float(qpos[7 + i]))
    task_env.robot.set_gripper(float(qpos[6]), "left")
    task_env.robot.set_gripper(float(qpos[13]), "right")
    for _ in range(settle_steps):
        task_env.scene.step()


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval_policy_dataset(
    task_name: str,
    TASK_ENV,
    args: dict[str, Any],
    model,
    records: list[dict[str, Any]],
    video_size: str | None = None,
    repo_id: str | None = None,
) -> tuple[int, int, int]:
    """
    Dataset-mode eval loop — one episode per record (fixed seed + initial_qpos).
    Returns (num_success, num_total, num_unstable_skipped).

    Interaction pattern is identical to RoboTwin's eval_policy.py (random mode):
        reset_func(model)
        while take_action_cnt < step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
    """
    import model2robotwin_interface as _policy

    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    clear_cache_freq = int(args.get("clear_cache_freq", 5))
    step_lim = _load_step_limit(task_name)
    args["eval_mode"] = True

    unstable_skipped = 0

    for now_id, record in enumerate(records):
        episode_tag = f"task={task_name}, episode_index={record['episode_index']}, seed={record['seed']}"
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=record["seed"], is_test=True, **args)
            TASK_ENV.step_lim = step_lim
            _set_robot_joints(TASK_ENV, record["initial_qpos"])
            TASK_ENV.set_instruction(instruction=record["instruction"])

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", video_size, "-framerate", "10", "-i", "-",
                        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            succ = False
            _policy.reset_model(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                _policy.eval(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()

            if succ:
                TASK_ENV.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")

            TASK_ENV.test_num += 1
            TASK_ENV.close_env(clear_cache=((now_id + 1) % clear_cache_freq == 0))

            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()

            print(
                f"\033[93m{task_name}\033[0m | \033[92m{repo_id or args['task_config']}\033[0m\n"
                f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
                f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
                f"episode_index: \033[90m{record['episode_index']}\033[0m "
                f"seed: \033[90m{record['seed']}\033[0m\n"
            )

        except UnStableError as exc:
            TASK_ENV.close_env()
            unstable_skipped += 1
            print(f"\033[93m[SKIP] {episode_tag}: UnStableError={exc}\033[0m")
            continue

        except Exception as exc:
            TASK_ENV.close_env()
            raise RuntimeError(f"{episode_tag}: error={exc}") from exc

    return TASK_ENV.suc, TASK_ENV.test_num, unstable_skipped
