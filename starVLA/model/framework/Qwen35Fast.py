from typing import List, Optional, Tuple, Any
import re
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
        self.use_state_input = bool(getattr(config.datasets.vla_data, "include_state", True))

        self.action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

    def _select_state_vector(self, state: Any) -> np.ndarray:
        """
        Convert state payload to a single state vector for prompt serialization.

        Training data may provide temporal state windows [T, D], while eval commonly
        provides a single vector [D]. We always serialize one timestep so train/infer
        conditioning stays consistent.
        """
        state_arr = np.asarray(state)
        if state_arr.ndim == 1:
            return state_arr
        if state_arr.ndim >= 2:
            # Align with the action anchor timestep used by chunked training windows.
            anchor = min(self.past_action_window_size, state_arr.shape[0] - 1)
            if state_arr.shape[0] > 1:
                print(f'[WARNING] data lodaer load more than 1 state. {state_arr.shape}, use index {anchor}')
            return state_arr[anchor].reshape(-1)
        raise ValueError(f"Unsupported state shape for prompt conditioning: {state_arr.shape}")

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
            instruction = example.get("lang", None)

            if self.use_state_input:
                state = self._select_state_vector(example["state"]) # NOTE: robot state should always provided when state input is enabled
                discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized))
                if instruction is not None: # For ablation study use.
                    prompt = f"{instruction}, State: {state_str}"
                else:
                    prompt = f"State: {state_str}"
            else:
                prompt = instruction if instruction is not None else ""

            # Keep fast-specific action prefix in framework prompt construction.
            instructions.append(f"{prompt}\nAction: ")

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
            instruction = example.get("lang", None)

            if self.use_state_input:
                state = self._select_state_vector(example["state"]) # NOTE: robot state should always provided when state input is enabled
                discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized))
                if instruction is not None: # For ablation study use.
                    prompt = f"{instruction}, State: {state_str}"
                else:
                    prompt = f"State: {state_str}"
            else:
                prompt = instruction if instruction is not None else ""

            # Keep fast-specific action prefix in framework prompt construction.
            instructions.append(f"{prompt}\nAction: ")

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        fast_proc = self.action_model.fast_tokenizer
        expected_t = int(fast_proc.time_horizon)
        expected_d = int(fast_proc.action_dim)

        # QWen35 forward pass
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        # Upper bound: BPE can only merge chars, never split, so token count <= T*D.
        # Add small slack for the "|" terminator and any edge cases.
        max_new_tokens = expected_t * expected_d + 16

        _tokenizer = self.qwen_vl_interface.processor.tokenizer
        _pipe_token_id = _tokenizer.convert_tokens_to_ids("|")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=[_tokenizer.eos_token_id, _pipe_token_id],
            )

        # Parse only newly generated tokens to avoid ambiguous anchor on prompt-side "Action: ".
        prompt_len = qwen_inputs["input_ids"].shape[1]
        generated_continuation = generated_ids[:, prompt_len:]
        expected_chars = expected_t * expected_d
        B = generated_continuation.shape[0]

        normalized_actions = np.zeros((B, expected_t, expected_d), dtype=np.float32)
        valid_timesteps = [0] * B

        from scipy.fft import idct as _idct

        sample_texts = {}  # b -> decoded text, for reporting in the all-zero warning
        for b in range(B):
            text = tokenizer.decode(generated_continuation[b].tolist(), skip_special_tokens=False)
            sample_texts[b] = text

            has_terminator = "|" in text
            search_region = text.split("|")[0] if has_terminator else text
            all_fast_ids = [
                int(m.group(1))
                for m in re.finditer(r'<robot_action_(\d+)>', search_region)
            ]
            if not all_fast_ids:
                logger.warning("Sample %d: no action tokens found. has_terminator=%s", b, has_terminator)
                continue

            try:
                # Full BPE decode once — avoids individual-vs-full-sequence length mismatch.
                dct_chars = fast_proc.bpe_tokenizer.decode(all_fast_ids)
                n_chars = len(dct_chars)

                if not has_terminator or n_chars >= expected_chars:
                    # Case 3 (no "|") or surplus chars: truncate to expected_chars.
                    actual_t = expected_t
                    dct_chars = dct_chars[:expected_chars]
                    if not has_terminator:
                        logger.warning(
                            "Sample %d: [Case 3] no '|', budget exhausted. "
                            "n_chars=%d → actual_t=%d. text=%r",
                            b, n_chars, actual_t, text,
                        )
                else:
                    # Case 2: model output "|" before horizon — early termination.
                    actual_t = n_chars // expected_d
                    if actual_t == 0:
                        logger.warning(
                            "Sample %d: [Case 2] only %d chars, too short for 1 timestep. text=%r",
                            b, n_chars, text,
                        )
                        continue
                    dct_chars = dct_chars[:actual_t * expected_d]
                    logger.warning(
                        "Sample %d: [Case 2] early termination — n_chars=%d → actual_t=%d (max %d). text=%r",
                        b, n_chars, actual_t, expected_t, text,
                    )

                # Decode DCT directly from the (already length-validated) char string.
                # Bypasses fast_proc.decode() entirely to avoid BPE re-decode and
                # the side-effect that mutates fast_proc.time_horizon.
                dct_vals = np.array(list(map(ord, dct_chars)), dtype=np.float64) + fast_proc.min_token
                dct_coeffs = dct_vals.reshape(actual_t, expected_d)
                action_arr = _idct(dct_coeffs / fast_proc.scale, axis=0, norm="ortho").astype(np.float32)

                # Pad short sequences with zeros so output is always (expected_t, expected_d).
                if actual_t < expected_t:
                    action_arr = np.pad(action_arr, ((0, expected_t - actual_t), (0, 0)))

                normalized_actions[b] = action_arr
                valid_timesteps[b] = actual_t

            except Exception as exc:
                logger.warning("Sample %d: FAST decode failed: %s. text=%r", b, exc, text)

        decode_zero_indices = [b for b in range(B) if np.allclose(normalized_actions[b], 0.0)]
        if decode_zero_indices:
            for b in decode_zero_indices:
                logger.warning(
                    "FAST action decode returned all-zero actions for sample %d/%d. text=%r",
                    b, B, sample_texts.get(b, "<unknown>"),
                )

        return {
            "normalized_actions": normalized_actions,
            "valid_timesteps": valid_timesteps,
        }

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on 
        """
        action_body = ''.join([f"<robot_action_{token}>" for token in tokens])
        # Action prefix is already part of prompt; solution only contains action body and terminator.
        return f"{action_body}|" # you should add <robot_action_{token}> to VLM as special tokens, 


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