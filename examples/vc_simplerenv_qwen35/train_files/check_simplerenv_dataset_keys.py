#!/usr/bin/env python3
"""Print actual modality keys from LeRobot datasets used by SimplerEnv/OXE.

Usage examples:
  python examples/vc_simplerenv_qwen35/train_files/check_simplerenv_dataset_keys.py
  python examples/vc_simplerenv_qwen35/train_files/check_simplerenv_dataset_keys.py \
    --data-root data/SimplerEnv/hf_lerobot \
    --datasets bridge_orig_1.0.0_lerobot fractal20220817_data_0.1.0_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _sorted_keys(d: dict[str, Any]) -> list[str]:
    return sorted(d.keys())


def _print_section(title: str, section: dict[str, Any]) -> None:
    print(f"  [{title}] ({len(section)} keys)")
    for key in _sorted_keys(section):
        value = section[key]
        if isinstance(value, dict) and "start" in value and "end" in value:
            print(f"    - {key}: start={value['start']}, end={value['end']}")
        elif isinstance(value, dict) and "original_key" in value:
            print(f"    - {key}: original_key={value['original_key']}")
        else:
            print(f"    - {key}: {value}")


def inspect_dataset(dataset_dir: Path) -> None:
    modality_path = dataset_dir / "meta" / "modality.json"
    print("=" * 80)
    print(f"Dataset: {dataset_dir.name}")
    print(f"Path: {dataset_dir}")

    if not modality_path.exists():
        print(f"  [ERROR] Missing file: {modality_path}")
        return

    with modality_path.open("r", encoding="utf-8") as f:
        modality = json.load(f)

    for name in ["state", "action", "video", "annotation"]:
        section = modality.get(name, {})
        if isinstance(section, dict):
            _print_section(name, section)
        else:
            print(f"  [{name}] unexpected type: {type(section).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SimplerEnv dataset modality keys")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/SimplerEnv/hf_lerobot"),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bridge_orig_1.0.0_lerobot", "fractal20220817_data_0.1.0_lerobot"],
        help="Dataset folder names under --data-root",
    )
    args = parser.parse_args()

    data_root = args.data_root
    print(f"Data root: {data_root.resolve()}")

    for name in args.datasets:
        inspect_dataset(data_root / name)


if __name__ == "__main__":
    main()
