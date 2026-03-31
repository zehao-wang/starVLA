"""
utils/eval_orchestrator.py — distributed dataset-mode eval orchestrator.

Spawns one worker process per GPU service port.  Workers share a task queue
(one item = all episodes for one task).  Results are collected and written
to a JSON summary.

Called by run_eval_distributed.sh:

    conda run -n RoboTwin python utils/eval_orchestrator.py \\
        --dataset-name  lerobot_robotwin_rand20k   \\
        --ports         5694,5695,5696,5697         \\
        --exp-name      starvla_dist_eval           \\
        --policy-config deploy_policy.yml

Dataset layout expected:
    HF_LEROBOT_HOME/<dataset_name>/Randomized/<task_name>/meta/env_init_states.jsonl

Requires PYTHONPATH to already include:
    packages/RoboTwin
    packages/starVLA
    eval_files/          (so "utils.*" and "model2robotwin_interface" resolve)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed eval orchestrator for starVLA / RoboTwin2")
    p.add_argument("--dataset-name",  required=True,  help="Top-level dataset name  (e.g. lerobot_robotwin_rand20k)")
    p.add_argument("--ports",         required=True,  help="Comma-separated policy server ports  (e.g. 5694,5695,5696,5697)")
    p.add_argument("--exp-name",      default="starvla_dist_eval")
    p.add_argument("--max-episodes",  default=None,
                   type=lambda v: None if v.strip().lower() in ("none", "null", "") else int(v),
                   help="Max episodes per task (default: all)")
    p.add_argument("--policy-config", required=True,  help="Path to deploy_policy.yml")
    p.add_argument("--output-root",   default=None,   help="Root dir for results (default: starVLA/results)")
    p.add_argument("--task-name",     default=None,   help="Limit eval to a single task subdirectory")
    p.add_argument("--gpu-ids",       default=None,   help="Comma-separated GPU IDs matching --ports order (e.g. 0,1,2,3)")
    return p.parse_args()


def _load_usr_args(policy_config: str) -> dict[str, Any]:
    eval_files_path = Path(__file__).resolve().parent.parent
    starvla_path = eval_files_path.parent.parent.parent

    with open(policy_config, "r", encoding="utf-8") as f:
        usr_args: dict[str, Any] = yaml.safe_load(f)

    ckpt = usr_args.get("policy_ckpt_path", "")
    if ckpt and not Path(ckpt).is_absolute():
        usr_args["policy_ckpt_path"] = str(starvla_path / ckpt)

    return usr_args


def _get_lerobot_home() -> Path:
    return Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot"),
        )
    )


def _discover_task_items(
    dataset_name: str,
    task_filter: str | None,
    load_metadata,     # callable: (repo_id) -> (init_states, episodes)
    build_records,     # callable: (init_states, episodes) -> records
) -> list[dict]:
    """
    Walk HF_LEROBOT_HOME/<dataset_name>/Randomized/ and build one task item
    per (task_name, task_config) group.  Each item carries its own repo_id.
    """
    dataset_root = _get_lerobot_home() / dataset_name / "Randomized"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    task_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if task_filter:
        task_dirs = [d for d in task_dirs if d.name == task_filter]
        if not task_dirs:
            raise ValueError(f"Task '{task_filter}' not found under {dataset_root}")

    task_items: list[dict] = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        repo_id = f"{dataset_name}/Randomized/{task_name}"
        init_states, episodes = load_metadata(repo_id)
        records = sorted(build_records(init_states, episodes), key=lambda r: r["episode_index"])

        grouped: dict[str, list] = defaultdict(list)
        for rec in records:
            grouped[rec["task_config"]].append(rec)

        for task_config, task_records in grouped.items():
            task_items.append({
                "task_name":   task_name,
                "task_config": task_config,
                "records":     task_records,
                "repo_id":     repo_id,
            })

    return task_items


def _print_banner(dataset_name, ports, exp_name, max_episodes, task_filter, task_items) -> None:
    print("=" * 72)
    print(f"  dataset       : {dataset_name}")
    print(f"  ports         : {ports}")
    print(f"  num_workers   : {len(ports)}")
    print(f"  exp_name      : {exp_name}")
    print(f"  max_episodes  : {max_episodes or '(all)'}")
    print(f"  task_filter   : {task_filter or '(all)'}")
    print(f"  task_queue    : {len(task_items)} task(s)")
    for item in task_items:
        print(f"    {item['task_name']} ({len(item['records'])} episodes)")
    print("=" * 72)
    print(flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Use 'spawn' to avoid issues with sapien / CUDA in forked processes
    mp.set_start_method("spawn", force=True)

    # RoboTwin uses relative paths (e.g. ./assets/…) that resolve from its root.
    # Must chdir before any `from envs import …` triggers module-level file I/O.
    robotwin_path = os.environ.get("ROBOTWIN_PATH", "")
    if robotwin_path:
        os.chdir(robotwin_path)
    else:
        print("WARNING: ROBOTWIN_PATH not set; relative-path imports from RoboTwin may fail.")

    args = _parse_args()

    eval_files_path = Path(__file__).resolve().parent.parent
    starvla_path = eval_files_path.parent.parent.parent

    usr_args = _load_usr_args(args.policy_config)
    ports = [int(p.strip()) for p in args.ports.split(",")]
    output_root = args.output_root or str(starvla_path / "results")

    # Dataset helpers — imported after chdir so envs module-level I/O finds ./assets/
    from utils.robotwin_helpers import _load_test_metadata, _build_episode_records

    task_items = _discover_task_items(
        dataset_name=args.dataset_name,
        task_filter=args.task_name,
        load_metadata=_load_test_metadata,
        build_records=_build_episode_records,
    )

    if not task_items:
        print("ERROR: no tasks found.")
        sys.exit(1)

    _print_banner(args.dataset_name, ports, args.exp_name, args.max_episodes, args.task_name, task_items)

    # ---------- queues ----------
    task_queue:   mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for item in task_items:
        task_queue.put(item)

    # ---------- launch workers ----------
    from utils.eval_worker import worker_main

    # Workers set CUDA_VISIBLE_DEVICES so Sapien/Vulkan rendering stays on the
    # same physical GPU as the policy server they talk to.
    gpu_ids = (
        [int(g.strip()) for g in args.gpu_ids.split(",")]
        if args.gpu_ids
        else list(range(len(ports)))
    )
    if len(gpu_ids) != len(ports):
        print("ERROR: --gpu-ids length must match --ports length")
        sys.exit(1)

    processes: list[mp.Process] = []
    for i, port in enumerate(ports):
        gpu_id = gpu_ids[i]
        p = mp.Process(
            target=worker_main,
            kwargs=dict(
                worker_id=i,
                port=port,
                gpu_id=gpu_id,
                task_queue=task_queue,
                result_queue=result_queue,
                usr_args=usr_args,
                output_root=output_root,
                exp_name=args.exp_name,
                max_episodes=args.max_episodes,
            ),
            daemon=True,
            name=f"eval-worker-{i}",
        )
        p.start()
        processes.append(p)
        print(f"[Orchestrator] Worker {i} started  PID={p.pid}  port={port}  CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)

    # ---------- collect results ----------
    results: list[dict] = []
    num_tasks = len(task_items)
    print(f"\n[Orchestrator] Waiting for {num_tasks} task result(s) …\n", flush=True)

    # exp-level dir for aggregate files (partial + final summary)
    exp_root = Path(output_root) / "robotwin2_dataset" / args.exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    # mapping task_name → repo_id so we can reconstruct the per-task folder
    task_name_to_repo_id: dict[str, str] = {
        item["task_name"]: item["repo_id"] for item in task_items
    }

    collected = 0
    while collected < num_tasks:
        # Check if all workers are dead — if so, no more results will arrive
        alive = [p for p in processes if p.is_alive()]
        if not alive and collected < num_tasks:
            print(
                f"[Orchestrator] WARNING: all workers exited but only {collected}/{num_tasks} "
                f"results received. Remaining tasks had no worker to process them.",
                flush=True,
            )
            break

        try:
            result = result_queue.get(timeout=5)
        except Exception:
            continue   # timeout — loop back and re-check alive workers

        results.append(result)
        collected += 1
        tn = result.get("task_name", "?")
        if "error" in result:
            print(f"  [ERROR ] {tn}: {result['error']}", flush=True)
        else:
            rate = result["success_rate"] * 100
            print(
                f"  [DONE  ] {tn}: "
                f"{result['success']}/{result['total']} = {rate:.1f}%  "
                f"(worker {result['worker_id']})",
                flush=True,
            )

        # Write per-task result into the same folder the worker uses for videos
        repo_id = task_name_to_repo_id.get(tn, tn)
        task_run_root = exp_root / repo_id.replace("/", "_")
        task_run_root.mkdir(parents=True, exist_ok=True)
        with open(task_run_root / "_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Update running partial summary at exp-level
        partial_ok = [r for r in results if "error" not in r]
        partial_fail = [r for r in results if "error" in r]
        p_success = sum(r["success"] for r in partial_ok)
        p_total   = sum(r["total"]   for r in partial_ok)
        partial_summary = {
            "dataset_name": args.dataset_name,
            "exp_name":     args.exp_name,
            "progress":     f"{collected}/{num_tasks}",
            "overall": {
                "success":      p_success,
                "total":        p_total,
                "success_rate": p_success / p_total if p_total > 0 else 0.0,
            },
            "tasks":  {r["task_name"]: r for r in partial_ok},
            "errors": {r["task_name"]: r["error"] for r in partial_fail},
        }
        with open(exp_root / "_partial_summary.json", "w", encoding="utf-8") as f:
            json.dump(partial_summary, f, ensure_ascii=False, indent=2)

    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    # ---------- summary ----------
    ok_results   = [r for r in results if "error" not in r]
    fail_results = [r for r in results if "error" in r]

    overall_success  = sum(r["success"]          for r in ok_results)
    overall_total    = sum(r["total"]             for r in ok_results)
    overall_planned  = sum(r["planned_total"]     for r in ok_results)
    overall_unstable = sum(r["unstable_skipped"]  for r in ok_results)
    overall_rate     = overall_success / overall_total if overall_total > 0 else 0.0

    summary = {
        "dataset_name": args.dataset_name,
        "exp_name":     args.exp_name,
        "ports":        ports,
        "overall": {
            "success":          overall_success,
            "total":            overall_total,
            "planned_total":    overall_planned,
            "unstable_skipped": overall_unstable,
            "success_rate":     overall_rate,
        },
        "tasks":  {r["task_name"]: r for r in ok_results},
        "errors": {r["task_name"]: r["error"] for r in fail_results},
    }

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = exp_root / f"_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    for r in ok_results:
        print(
            f"{r['task_name']:<36} "
            f"{r['success']}/{r['total']} = {r['success_rate'] * 100:.1f}%"
            f"  (planned={r['planned_total']}, unstable_skipped={r['unstable_skipped']})"
        )
    for r in fail_results:
        print(f"{r['task_name']:<36} ERROR: {r['error']}")
    print(
        f"{'TOTAL':<36} "
        f"{overall_success}/{overall_total} = {overall_rate * 100:.1f}%"
        f"  (planned={overall_planned}, unstable_skipped={overall_unstable})"
    )
    print(f"Saved: {summary_path}")
    # Clean up partial summary now that the final one is written
    partial_path = exp_root / "_partial_summary.json"
    if partial_path.exists():
        partial_path.unlink()
    print("=" * 72)


if __name__ == "__main__":
    main()
