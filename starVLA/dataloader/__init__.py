import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _log_vla_dataset_stats(vla_dataset) -> None:
    """Log dataset usage stats (episodes/steps) for easier training diagnostics."""
    if not _is_rank0():
        return

    try:
        if hasattr(vla_dataset, "datasets") and isinstance(vla_dataset.datasets, list):
            total_effective_steps = 0
            total_raw_steps = 0
            total_total_traj = 0
            total_kept_traj = 0
            total_skipped_traj = 0
            logger.info("[DATA STATS] Per-subdataset statistics:")

            for sub_ds in vla_dataset.datasets:
                ds_name = getattr(sub_ds, "dataset_name", type(sub_ds).__name__)
                effective_steps = len(sub_ds) if hasattr(sub_ds, "__len__") else 0
                total_traj = int(getattr(sub_ds, "_total_trajectories", len(getattr(sub_ds, "trajectory_ids", []))))
                kept_traj = int(getattr(sub_ds, "_processed_trajectories", total_traj))
                skipped_traj = int(getattr(sub_ds, "_skipped_trajectories", max(total_traj - kept_traj, 0)))
                raw_steps = int(getattr(sub_ds, "_raw_total_steps", effective_steps))

                total_effective_steps += int(effective_steps)
                total_raw_steps += int(raw_steps)
                total_total_traj += int(total_traj)
                total_kept_traj += int(kept_traj)
                total_skipped_traj += int(skipped_traj)
                logger.info(
                    f"[DATA STATS]   {ds_name}: trajectories(total={total_traj}, usable={kept_traj}, skipped={skipped_traj}), "
                    f"steps(total={raw_steps}, usable={effective_steps})"
                )

            logger.info(
                f"[DATA STATS] Mixture total: trajectories(total={total_total_traj}, usable={total_kept_traj}, skipped={total_skipped_traj}), "
                f"steps(total={total_raw_steps}, usable={total_effective_steps})"
            )
        else:
            steps = len(vla_dataset) if hasattr(vla_dataset, "__len__") else -1
            logger.info(f"[DATA STATS] Dataset steps={steps}")
    except Exception as exc:
        logger.warning(f"[DATA STATS] Failed to compute dataset stats: {exc}")

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        output_dir = Path(cfg.output_dir)
        stats_path = output_dir / "dataset_statistics.json"
        cached_statistics_path = stats_path if stats_path.exists() else None
        if cached_statistics_path is not None:
            logger.info(f"Found existing dataset_statistics.json, loading from cache: {cached_statistics_path}")
        else:
            logger.info("No cached dataset_statistics.json found, will compute and save.")

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, cached_statistics_path=cached_statistics_path)
        _log_vla_dataset_stats(vla_dataset)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )
        if _is_rank0() and cached_statistics_path is None:
            vla_dataset.save_dataset_statistics(stats_path)
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
