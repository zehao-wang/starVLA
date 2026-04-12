# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen35-GR00T Framework
Qwen3.5-VL + GR00T flow-matching head for continuous action prediction.
"""
from typing import List, Optional, Tuple
import os
import torch
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Qwen35GR00T")
class Qwen35_GR00T(baseframework):
    """
    Pure GR00T objective on top of Qwen3.5-VL:
      - single branch V+L conditioning
      - flow-matching action loss only
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align action-head cross-attention input dim with VLM hidden size.
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.use_state_input = bool(getattr(config.datasets.vla_data, "include_state", False))
        logger.info(f"[Qwen35GR00T] include_state (use_state_input) = {self.use_state_input}")
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = None
        if self.use_state_input and "state" in examples[0]:
            state = [example["state"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            required_horizon = self.future_action_window_size + 1
            if actions_t.shape[1] < required_horizon:
                raise ValueError(
                    f"actions length {actions_t.shape[1]} is shorter than required horizon {required_horizon}."
                )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state_t = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_dit_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        state = None
        if self.use_state_input and "state" in examples[0]:
            state = [example["state"] for example in examples]

        debug_state_shape = os.getenv("STARVLA_DEBUG_STATE_SHAPE", "0").lower() in {"1", "true", "yes", "on"}
        if debug_state_shape:
            raw_shape = np.array(state).shape if state is not None else None
            print(f"[Qwen35GR00T.predict_action] raw state shape: {raw_shape}")

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state_t = None
        if state is not None:
            state_np = np.array(state)
            if state_np.ndim == 1:
                state_np = state_np[None, None, :]
            elif state_np.ndim == 2:
                state_np = state_np[:, None, :]
            elif state_np.ndim != 3:
                raise ValueError(f"Unsupported state shape {state_np.shape}, expected [B, D] or [B, 1, D].")

            if debug_state_shape:
                print(f"[Qwen35GR00T.predict_action] normalized state shape: {state_np.shape}")

            state_t = torch.from_numpy(state_np).to(last_hidden.device, dtype=last_hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state_t)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    model: Qwen35_GR00T = Qwen35_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    horizon = cfg.framework.action_model.future_action_window_size + 1
    sample = {
        "action": np.random.uniform(-1, 1, size=(horizon, cfg.framework.action_model.action_dim)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model(batch)
    print(f"Action Loss: {out['action_dit_loss'].item()}")

    pred = model.predict_action([sample])
    print(f"Pred shape: {pred['normalized_actions'].shape}")
