from typing import List
from typing import List, Optional, Tuple, Any
import torch
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2026-04-01.
####################################################

@FRAMEWORK_REGISTRY.register("Qwen35Fast")
class Qwen35_Fast(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen3.5 interface for fused language/vision token embeddings
      
    Focus: Predict future discretized actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config) # NOTE: base_vlm must contain Qwen3.5
        self.action_model = get_action_model(None)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_ar_loss
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        actions = [example["action"] for example in examples]  # label [B， len, action_dim]

        instructions = []
        for example in examples:
            state = example["state"].flatten() # NOTE: robot state should always provided
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized))

            instruction = example.get("lang", None)
            if instruction is not None: # For ablation study use.
                prompt = f"{instruction}, State: {state_str}"
            else:
                prompt = f"State: {state_str}"
            instructions.append(prompt)

        # step 0: map_raw_action_to_vlm_action
        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)  # List[str]
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        # QWen35 forward pass
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions,  solutions=vlm_action_tokens)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss):
            raise RuntimeError(
                "action_loss is None or NaN. "
                "Check that the model has action special tokens added and labels are not fully masked. "
                "See starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
            )

        return {"action_ar_loss": vlm_action_loss}

    @torch.inference_mode()
    def predict_action( 
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]

        instructions = []
        for example in examples:
            state = example["state"].flatten() # NOTE: robot state should always provided
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized))

            instruction = example.get("lang", None)
            if instruction is not None: # For ablation study use.
                prompt = f"{instruction}, State: {state_str}"
            else:
                prompt = f"State: {state_str}"
            instructions.append(prompt)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # QWen35 forward pass
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        time_horizon = self.action_model.fast_tokenizer.time_horizon
        action_dim = self.action_model.fast_tokenizer.action_dim
        max_new_tokens = time_horizon * action_dim  # worst-case: 1 BPE token per DCT char
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=max_new_tokens,
            )
        # --- Extract and decoder vlm_action to continue actions ---
        # --- extrace token (index based on VLM) ---
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        # --- map index to fast tokenizer index space ---
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        # --- decode fast tokenizer index to action semantic ---
        normalized_actions = self.action_model.fast_tokenizer.decode(batch_fast_action_token_idx)

        return {"normalized_actions": normalized_actions}

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep all tokens falling within [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] in order of appearance.
        You may change it to "take only the first occurrence followed by continuous segment" as needed.
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # all action tokens
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            fast_ids = [t - act_min for t in seq]
            
            batch_fast_token_ids.append(fast_ids)
        
        return batch_fast_token_ids

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on 
        """
        return ''.join([f"<robot_action_{token}>" for token in tokens]) # you should add <robot_action_{token}> to VLM as special tokens, 


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/vc_Robotwin2_qwen35/train_files/starvla_cotrain_robotwin_qwen35_fast.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    # try get model
    model = Qwen35_Fast(cfg)
    print(model)

    # fake sample: action_dim=14, time_horizon=16, 3 views (cam_high, cam_left_wrist, cam_right_wrist)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16),  # (time_horizon, action_dim)
        "image": [image, image, image],                                          # 3 views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(14,)).astype(np.float32),       # (state_dim,)
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16),
        "image": [image, image, image],
        "lang": "The fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(14,)).astype(np.float32),
    }

    batch = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_ar_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action. for new model, it didn't learn to predict action token, so you would meet empty action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Normalized Action: {normalized_actions}")

    # test with dataloader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        break
    model(batch)
    action = model.predict_action([batch[0]])