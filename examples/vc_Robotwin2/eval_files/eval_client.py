"""
eval_client.py  —  Dataset-mode evaluation for starVLA on RoboTwin2.

Mirrors eval_policy.py structure. The only difference: env is initialized from
a LeRobot dataset (fixed seed + initial qpos) instead of random scene rollout.

Usage (via eval.sh):
    MODE=dataset bash eval.sh <repo_id> <task_name> [exp_name] [max_episodes] [gpu_id]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EVAL_FILES_PATH = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Imports that require PYTHONPATH (set by eval.sh)
# ---------------------------------------------------------------------------
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
import model2robotwin_interface as _policy


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
                pass  # fall through to init_states fallback below

    # Fallback: derive episodes from init_states (episode_index = line number, instruction from task_name)
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
# Eval loop (mirrors eval_policy() in eval_policy.py)
# ---------------------------------------------------------------------------

def eval_policy_dataset(task_name, TASK_ENV, args, model, records, video_size=None, repo_id=None):
    """
    Dataset-mode eval loop.  Mirrors eval_policy() from eval_policy.py.
    Instead of iterating random seeds until test_num successes, iterates over
    dataset records with fixed seed + initial_qpos per episode.
    """
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


# ---------------------------------------------------------------------------
# Main (mirrors main() in eval_policy.py)
# ---------------------------------------------------------------------------

def main(usr_args: dict[str, Any]) -> None:
    repo_id = usr_args["repo_id"]
    task_name_filter = usr_args.get("task_name")
    max_episodes = usr_args.get("max_episodes")
    exp_name = usr_args.get("exp_name", "starvla_dataset_eval")
    output_root = usr_args.get("output_root", "eval_result")
    policy_name = usr_args.get("policy_name", "model2robotwin_interface")

    print("=" * 72)
    print(f"  repo_id       : {repo_id}")
    print(f"  task_name     : {task_name_filter or '(all)'}")
    print(f"  max_episodes  : {max_episodes or '(all)'}")
    print(f"  exp_name      : {exp_name}")
    print("=" * 72)

    init_states, episodes = _load_test_metadata(repo_id)
    records = sorted(_build_episode_records(init_states, episodes), key=lambda r: r["episode_index"])

    if task_name_filter is not None:
        records = [r for r in records if r["task_name"] == task_name_filter]
        if not records:
            raise ValueError(f"No records for task_name={task_name_filter!r} in repo_id={repo_id!r}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["task_name"], record["task_config"])].append(record)

    run_root = Path(output_root) / "robotwin2_dataset" / exp_name / repo_id.replace("/", "_")
    run_root.mkdir(parents=True, exist_ok=True)

    model = _policy.get_model(usr_args)

    overall_success = 0
    overall_total = 0
    overall_planned = 0
    overall_unstable = 0
    summary: dict[str, dict[str, Any]] = {}

    for (task_name, task_config), task_records in grouped.items():
        if max_episodes is not None:
            task_records = task_records[:max(0, max_episodes)]

        args = build_env_args(task_name, task_config, policy_name)

        video_size = None
        if args.get("eval_video_log", False):
            video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"
            args["eval_video_save_dir"] = str(run_root)

        TASK_ENV = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()

        suc, total, unstable = eval_policy_dataset(task_name, TASK_ENV, args, model, task_records, video_size, repo_id)

        planned = len(task_records)
        rate = suc / total if total > 0 else 0.0
        summary[task_name] = {
            "task_config": task_config,
            "success": suc,
            "total": total,
            "planned_total": planned,
            "unstable_skipped": unstable,
            "success_rate": rate,
        }
        overall_success += suc
        overall_total += total
        overall_planned += planned
        overall_unstable += unstable

        with open(run_root / "_result.txt", "w", encoding="utf-8") as f:
            f.write(f"task={task_name}\ntask_config={task_config}\n")
            f.write(f"repo_id={repo_id}\nexp_name={exp_name}\n")
            f.write(f"success={suc}\nplanned_total={planned}\n")
            f.write(f"total={total}\nunstable_skipped={unstable}\n")
            f.write(f"success_rate={rate:.6f}\n")

    overall_rate = overall_success / overall_total if overall_total > 0 else 0.0
    global_summary = {
        "repo_id": repo_id,
        "exp_name": exp_name,
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "planned_total": overall_planned,
            "unstable_skipped": overall_unstable,
            "success_rate": overall_rate,
        },
        "tasks": summary,
    }

    summary_path = run_root / f"_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    for tn, item in summary.items():
        print(
            f"{tn:<36} {item['success']}/{item['total']} = {item['success_rate'] * 100:.1f}%"
            f"  (planned={item['planned_total']}, unstable_skipped={item['unstable_skipped']})"
        )
    print(
        f"{'TOTAL':<36} {overall_success}/{overall_total} = {overall_rate * 100:.1f}%"
        f"  (planned={overall_planned}, unstable_skipped={overall_unstable})"
    )
    print(f"Saved summary : {summary_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args_and_config():
    def _optional_int(value: str) -> int | None:
        if value.strip().lower() in {"none", "null", ""}:
            return None
        return int(value)

    parser = argparse.ArgumentParser(
        description="Dataset-mode eval: reproduce LeRobot dataset episodes on RoboTwin2."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--max-episodes", type=_optional_int, default=None)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--policy-name", default="model2robotwin_interface")
    args = parser.parse_args()

    policy_config_path = args.policy_config or str(EVAL_FILES_PATH / "deploy_policy.yml")
    with open(policy_config_path, "r", encoding="utf-8") as f:
        usr_args: dict[str, Any] = yaml.safe_load(f)

    # Resolve relative paths in config against STARVLA_PATH (eval_files/../../..)
    starvla_path = EVAL_FILES_PATH.parent.parent.parent
    ckpt = usr_args.get("policy_ckpt_path", "")
    if ckpt and not Path(ckpt).is_absolute():
        usr_args["policy_ckpt_path"] = str(starvla_path / ckpt)

    usr_args["repo_id"] = args.repo_id
    usr_args["task_name"] = args.task_name
    usr_args["max_episodes"] = args.max_episodes
    usr_args["exp_name"] = args.exp_name
    usr_args["output_root"] = args.output_root or str(starvla_path / "results")
    usr_args["policy_name"] = args.policy_name

    return usr_args


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
