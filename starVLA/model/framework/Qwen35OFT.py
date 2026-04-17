# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Qwen35-OFT Framework

Qwen3.5 (natively multimodal) + MLP action head (L1 regression) for continuous action prediction.
Inspired by OpenVLA-OFT: injects special action tokens into the instruction,
extracts the corresponding hidden states from the VLM, and regresses continuous
actions via an MLP ResNet head.

Key differences from QwenOFT:
  - Uses Qwen3.5 backbone instead of Qwen2.5-VL
  - Supports optional state input
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("Qwen35OFT")
class Qwen35_OFT(baseframework):
    """
    Qwen3.5 + OFT-style MLP action head.

    Approach:
      - Inject action placeholder tokens (🔍 × chunk_len) into the instruction
      - Run VLM forward to get contextualised hidden states
      - Gather hidden states at action token positions
      - Predict continuous actions via MLP (L1 loss)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align action_hidden_dim with VLM hidden size
        config.framework.action_model.action_hidden_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.use_state_input = bool(getattr(config.datasets.vla_data, "include_state", False))
        logger.info(f"[Qwen35OFT] include_state (use_state_input) = {self.use_state_input}")

        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        self.l1_loss = nn.L1Loss()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # Append action placeholder tokens to instruction
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [inst + prompt_suffix for inst in instructions]

        # VLM forward
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Action prediction + L1 loss
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # [B, chunk_len, action_dim]

            actions_t = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Append action placeholder tokens
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [inst + prompt_suffix for inst in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,
    ) -> torch.Tensor:
        """
        Extract the last `chunk_len` action-token hidden states from each sample.
        Vectorised over the batch dimension.
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Action token count < {self.chunk_len} for samples: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len]
        selected_pos = topk_pos.sort(dim=-1).values                  # [B, chunk_len]

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    model = Qwen35_OFT(cfg)
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
    print(f"Action Loss: {out['action_loss'].item()}")

    pred = model.predict_action([sample])
    print(f"Pred shape: {pred['normalized_actions'].shape}")
