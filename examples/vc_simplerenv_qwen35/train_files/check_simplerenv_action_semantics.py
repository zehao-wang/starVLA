#!/usr/bin/env python3
"""Heuristic check for action semantics (delta-like vs absolute-like) on LeRobot OXE datasets.

This script reads parquet episodes and compares action against state:
- abs_gap:   |action_t - state_t|
- delta_gap: |action_t - (state_{t+1} - state_t)|

If delta_gap << abs_gap, actions look delta-like.
If abs_gap << delta_gap, actions look absolute-like.

Usage:
  python examples/vc_simplerenv_qwen35/train_files/check_simplerenv_action_semantics.py
  python examples/vc_simplerenv_qwen35/train_files/check_simplerenv_action_semantics.py --max-episodes 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_DEFAULTS = [
    "bridge_orig_1.0.0_lerobot",
    "fractal20220817_data_0.1.0_lerobot",
]

ACTION_VECTOR_CANDIDATES = ["action", "actions", "observation.action", "agent.action"]
STATE_VECTOR_CANDIDATES = ["state", "states", "observation.state", "robot_state", "proprio", "observation.proprio"]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_1d_float_array(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        arr = v.astype(np.float64, copy=False).reshape(-1)
        return arr
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v, dtype=np.float64).reshape(-1)
        except Exception:
            return None
    if np.isscalar(v):
        try:
            return np.asarray([v], dtype=np.float64)
        except Exception:
            return None
    return None


def _pick_first_present(columns: list[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _find_dotted_pairs(columns: list[str], shared_keys: list[str]) -> list[tuple[str, str]]:
    pairs = []
    colset = set(columns)
    for k in shared_keys:
        a = f"action.{k}"
        s = f"state.{k}"
        if a in colset and s in colset:
            pairs.append((a, s))
    return pairs


def _extract_from_vector(vec: Any, ranges: dict[str, dict[str, int]], keys: list[str]) -> np.ndarray | None:
    arr = _as_1d_float_array(vec)
    if arr is None:
        return None
    out = []
    for k in keys:
        spec = ranges.get(k)
        if not isinstance(spec, dict) or "start" not in spec or "end" not in spec:
            return None
        start = int(spec["start"])
        end = int(spec["end"])
        if start < 0 or end <= start or end > arr.shape[0]:
            return None
        out.append(arr[start:end])
    if not out:
        return None
    return np.concatenate(out).astype(np.float64, copy=False)


def _episode_mats(df: pd.DataFrame, action_ranges: dict, state_ranges: dict, shared_keys: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    cols = [str(c) for c in df.columns.tolist()]

    dotted_pairs = _find_dotted_pairs(cols, shared_keys)
    if dotted_pairs:
        a_rows = []
        s_rows = []
        for i in range(len(df)):
            a_vals = []
            s_vals = []
            ok = True
            for a_col, s_col in dotted_pairs:
                try:
                    a = float(df[a_col].iloc[i])
                    s = float(df[s_col].iloc[i])
                except Exception:
                    ok = False
                    break
                if not np.isfinite(a) or not np.isfinite(s):
                    ok = False
                    break
                a_vals.append(a)
                s_vals.append(s)
            if ok:
                a_rows.append(a_vals)
                s_rows.append(s_vals)

        if len(a_rows) >= 2:
            return np.asarray(a_rows, dtype=np.float64), np.asarray(s_rows, dtype=np.float64)

    a_col = _pick_first_present(cols, ACTION_VECTOR_CANDIDATES)
    s_col = _pick_first_present(cols, STATE_VECTOR_CANDIDATES)
    if a_col is None or s_col is None:
        return None

    a_rows = []
    s_rows = []
    for i in range(len(df)):
        a = _extract_from_vector(df[a_col].iloc[i], action_ranges, shared_keys)
        s = _extract_from_vector(df[s_col].iloc[i], state_ranges, shared_keys)
        if a is None or s is None or a.shape != s.shape:
            continue
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(s)):
            continue
        a_rows.append(a)
        s_rows.append(s)

    if len(a_rows) < 2:
        return None
    return np.stack(a_rows, axis=0), np.stack(s_rows, axis=0)


def analyze_dataset(root: Path, dataset_name: str, max_episodes: int) -> None:
    dataset_dir = root / dataset_name
    modality_path = dataset_dir / "meta" / "modality.json"

    print("=" * 80)
    print(f"Dataset: {dataset_name}")

    if not modality_path.exists():
        print(f"  [ERROR] Missing modality file: {modality_path}")
        return

    modality = _read_json(modality_path)
    action_ranges = modality.get("action", {}) if isinstance(modality.get("action", {}), dict) else {}
    state_ranges = modality.get("state", {}) if isinstance(modality.get("state", {}), dict) else {}
    shared_keys = sorted(set(action_ranges.keys()).intersection(state_ranges.keys()))
    if not shared_keys:
        print("  [ERROR] No shared action/state keys in modality.json")
        return

    files = sorted(dataset_dir.glob("data/*/*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("*.parquet"))
    files = files[:max_episodes]

    print(f"Episode files sampled: {len(files)}")
    print(f"Shared keys used for comparison: {shared_keys}")

    if not files:
        print("  [ERROR] No parquet files found")
        return

    abs_gaps = []
    delta_gaps = []
    used = 0

    for fp in files:
        df = pd.read_parquet(fp)
        mats = _episode_mats(df, action_ranges, state_ranges, shared_keys)
        if mats is None:
            continue
        a_mat, s_mat = mats
        if a_mat.shape[0] < 2:
            continue

        abs_gap = float(np.mean(np.abs(a_mat[:-1] - s_mat[:-1])))
        delta_state = s_mat[1:] - s_mat[:-1]
        delta_gap = float(np.mean(np.abs(a_mat[:-1] - delta_state)))

        if np.isfinite(abs_gap) and np.isfinite(delta_gap):
            abs_gaps.append(abs_gap)
            delta_gaps.append(delta_gap)
            used += 1

    if not abs_gaps:
        print("  [ERROR] No usable episodes after filtering")
        print("  Hint: run once with --debug-columns to inspect raw parquet column names")
        return

    abs_med = float(np.median(abs_gaps))
    delta_med = float(np.median(delta_gaps))
    ratio = delta_med / (abs_med + 1e-12)

    print(f"Used episodes: {used}")
    print(f"  median(|a_t - s_t|):                {abs_med:.6f}")
    print(f"  median(|a_t - (s_t+1 - s_t)|):      {delta_med:.6f}")
    print(f"  ratio delta_gap / abs_gap:          {ratio:.4f}")

    if ratio < 0.8:
        verdict = "delta-like (action resembles state increment)"
    elif ratio > 1.25:
        verdict = "absolute-like (action resembles state value)"
    else:
        verdict = "ambiguous/mixed"

    print(f"  heuristic verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OXE action semantics from parquet samples")
    parser.add_argument("--data-root", type=Path, default=Path("data/SimplerEnv/hf_lerobot"))
    parser.add_argument("--max-episodes", type=int, default=40)
    args = parser.parse_args()

    print(f"Data root: {args.data_root.resolve()}")

    for dataset_name in DATASET_DEFAULTS:
        analyze_dataset(args.data_root, dataset_name, args.max_episodes)


if __name__ == "__main__":
    main()
