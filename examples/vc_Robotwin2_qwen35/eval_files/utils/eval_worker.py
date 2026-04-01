"""
utils/eval_worker.py — worker process for distributed dataset-mode eval.

Each worker is bound to one policy server port.  It drains a shared
multiprocessing.Queue of TaskItems, running all episodes for each task
before pulling the next item.

A TaskItem is a dict:
    {
        "task_name":   str,
        "task_config": str,
        "records":     list[dict],   # episode records (seed, qpos, …)
    }

Results pushed to result_queue:
    {
        "task_name":        str,
        "task_config":      str,
        "success":          int,
        "total":            int,
        "planned_total":    int,
        "unstable_skipped": int,
        "success_rate":     float,
        "worker_id":        int,
    }
  or on unrecoverable error:
    {
        "task_name":  str,
        "task_config": str,
        "error":      str,
        "worker_id":  int,
    }
"""
from __future__ import annotations

import importlib
import queue as _queue_mod
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# worker_main — entry point called by multiprocessing.Process
# ---------------------------------------------------------------------------

def worker_main(
    worker_id: int,
    port: int,
    gpu_id: int,
    task_queue,       # multiprocessing.Queue[TaskItem]
    result_queue,     # multiprocessing.Queue[ResultItem]
    usr_args: dict[str, Any],
    output_root: str,
    exp_name: str,
    max_episodes: int | None,
) -> None:
    """
    Runs inside a subprocess (spawn).  Imports happen here so each process
    gets its own copy of all C-extension state (sapien physics, CUDA ctx, …).
    """
    import os
    # Pin this worker to its GPU so Sapien/Vulkan rendering stays off GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # RoboTwin uses relative paths (e.g. ./assets/…) — must chdir before any envs import.
    robotwin_path = os.environ.get("ROBOTWIN_PATH", "")
    if robotwin_path:
        os.chdir(robotwin_path)

    # Deferred imports — executed inside the subprocess after spawn
    import model2robotwin_interface as _policy_module
    from utils.robotwin_helpers import build_env_args, eval_policy_dataset

    tag = f"[Worker {worker_id} | port {port}]"
    print(f"{tag} Starting.", flush=True)

    # Build model client pointing at this worker's server port
    worker_args = {**usr_args, "port": port}
    try:
        model = _policy_module.get_model(worker_args)
    except Exception as exc:
        print(f"{tag} FATAL: could not initialise model client: {exc}", flush=True)
        # Drain the queue, pushing errors for every pending task
        while True:
            try:
                item = task_queue.get(timeout=2)
                result_queue.put({
                    "task_name": item["task_name"],
                    "task_config": item["task_config"],
                    "error": f"model init failed: {exc}",
                    "worker_id": worker_id,
                })
            except _queue_mod.Empty:
                break
        return

    while True:
        try:
            item = task_queue.get(timeout=10)
        except _queue_mod.Empty:
            break  # No more tasks

        task_name = item["task_name"]
        task_config = item["task_config"]
        records = item["records"]
        repo_id = item["repo_id"]

        if max_episodes is not None:
            records = records[: max(0, max_episodes)]

        print(f"{tag} Task: {task_name} ({len(records)} episodes)", flush=True)

        try:
            env_args = build_env_args(task_name, task_config, "model2robotwin_interface")

            video_size = None
            if env_args.get("eval_video_log", False):
                out_root = Path(output_root) / "robotwin2_dataset" / exp_name / repo_id.replace("/", "_")
                out_root.mkdir(parents=True, exist_ok=True)
                env_args["eval_video_save_dir"] = str(out_root)
                video_size = f"{env_args['head_camera_w']}x{env_args['head_camera_h']}"

            TASK_ENV = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()

            suc, total, unstable = eval_policy_dataset(
                task_name, TASK_ENV, env_args, model, records, video_size, repo_id
            )

            result_queue.put({
                "task_name": task_name,
                "task_config": task_config,
                "success": suc,
                "total": total,
                "planned_total": len(records),
                "unstable_skipped": unstable,
                "success_rate": suc / total if total > 0 else 0.0,
                "worker_id": worker_id,
            })
            print(
                f"{tag} Done: {task_name}  {suc}/{total} = "
                f"{suc / total * 100:.1f}%" if total > 0 else f"{tag} Done: {task_name}  0/0",
                flush=True,
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_queue.put({
                "task_name": task_name,
                "task_config": task_config,
                "error": str(exc),
                "worker_id": worker_id,
            })
            print(f"{tag} ERROR on {task_name}: {exc}", flush=True)

    print(f"{tag} Exiting — queue exhausted.", flush=True)
