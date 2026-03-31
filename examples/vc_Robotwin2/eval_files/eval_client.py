"""
eval_client.py  —  Single-GPU dataset-mode evaluation for starVLA on RoboTwin2.

Mirrors eval_policy.py structure. The only difference: env is initialized from
a LeRobot dataset (fixed seed + initial qpos) instead of random scene rollout.

Usage (via eval.sh):
    MODE=dataset bash eval.sh <repo_id> <task_name> [exp_name] [max_episodes] [gpu_id]

For multi-GPU distributed eval, use run_eval_distributed.sh instead.
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EVAL_FILES_PATH = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared helpers (dataset loading, env building, eval loop)
# ---------------------------------------------------------------------------
import model2robotwin_interface as _policy

from utils.robotwin_helpers import (
    _load_test_metadata,
    _build_episode_records,
    build_env_args,
    eval_policy_dataset,
)


# ---------------------------------------------------------------------------
# Main
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
        description="Single-GPU dataset-mode eval: reproduce LeRobot dataset episodes on RoboTwin2."
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
