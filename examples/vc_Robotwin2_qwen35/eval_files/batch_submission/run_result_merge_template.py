#!/usr/bin/env python3
"""Merge per-task Robotwin evaluation results from parallel sbatch jobs.

This script scans one experiment directory under:
  results/robotwin2_dataset/<exp_name>/
and merges all per-task "_result.json" files into one summary JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


def _infer_repo_root(script_path: Path) -> Path:
    # Typical layout: <repo>/results/Checkpoints/<run_id>/run_result_merge.py
    # So script_path.parents[3] points to <repo>.
    try:
        return script_path.parents[3]
    except IndexError:
        return Path.cwd()


def _parse_exp_name_from_deploy(deploy_path: Path) -> str | None:
    if not deploy_path.is_file():
        return None
    for line in deploy_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("exp_name:"):
            _, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def _parse_args(default_results_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Robotwin per-task eval results")
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name under results/robotwin2_dataset (default: read from deploy_policy.yml)",
    )
    parser.add_argument(
        "--results-root",
        default=str(default_results_root),
        help="Root directory containing robotwin2_dataset results",
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Direct path to one experiment directory (overrides --results-root/--exp-name)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional output summary json path (default: <exp_dir>/_merged_summary_<timestamp>.json)",
    )
    return parser.parse_args()


def _load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root = _infer_repo_root(script_path)
    default_results_root = repo_root / "results" / "robotwin2_dataset"

    args = _parse_args(default_results_root)

    deploy_path = script_dir / "deploy_policy.yml"
    deploy_exp_name = _parse_exp_name_from_deploy(deploy_path)

    if args.exp_dir:
        exp_dir = Path(args.exp_dir).resolve()
        exp_name = exp_dir.name
    else:
        if not (args.exp_name or deploy_exp_name):
            print("ERROR: exp name not found. Set --exp-name or ensure deploy_policy.yml has exp_name.")
            sys.exit(1)
        exp_name = args.exp_name or deploy_exp_name
        results_root = Path(args.results_root).resolve()
        exp_dir = results_root / exp_name

    if not exp_dir.is_dir():
        print(f"ERROR: experiment directory not found: {exp_dir}")
        sys.exit(1)

    result_files = sorted(exp_dir.rglob("_result.json"))
    if not result_files:
        print(f"ERROR: no _result.json found under {exp_dir}")
        print("Hint: run this after all eval sbatch jobs have finished.")
        sys.exit(1)

    ok_results: dict[str, dict] = {}
    error_results: dict[str, str] = {}
    duplicates: list[str] = []

    for result_file in result_files:
        result = _load_result(result_file)
        task_name = result.get("task_name") or result_file.parent.name

        if task_name in ok_results or task_name in error_results:
            duplicates.append(task_name)

        if "error" in result:
            error_results[task_name] = str(result["error"])
        else:
            ok_results[task_name] = result

    overall_success = sum(int(r.get("success", 0)) for r in ok_results.values())
    overall_total = sum(int(r.get("total", 0)) for r in ok_results.values())
    overall_planned = sum(int(r.get("planned_total", 0)) for r in ok_results.values())
    overall_unstable = sum(int(r.get("unstable_skipped", 0)) for r in ok_results.values())
    overall_rate = (overall_success / overall_total) if overall_total > 0 else 0.0

    summary = {
        "exp_name": exp_name,
        "exp_dir": str(exp_dir),
        "result_files": len(result_files),
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "planned_total": overall_planned,
            "unstable_skipped": overall_unstable,
            "success_rate": overall_rate,
        },
        "tasks": ok_results,
        "errors": error_results,
        "duplicates": sorted(set(duplicates)),
    }

    if args.save_path:
        out_path = Path(args.save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = exp_dir / f"_merged_summary_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"Experiment dir : {exp_dir}")
    print(f"Result files   : {len(result_files)}")
    print(f"OK tasks       : {len(ok_results)}")
    print(f"Error tasks    : {len(error_results)}")
    if duplicates:
        print(f"Duplicates     : {', '.join(sorted(set(duplicates)))}")
    print(f"TOTAL          : {overall_success}/{overall_total} = {overall_rate * 100:.1f}%")
    print(f"Saved          : {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()