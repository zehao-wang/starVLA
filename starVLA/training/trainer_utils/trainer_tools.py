"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
import re
import json
import numpy as np
import torch

from accelerate.logging import get_logger

logger = get_logger(__name__)


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters (both explicit freeze_modules and requires_grad=False e.g. LoRA base weights)
            params = [p for p in module.parameters() if id(p) not in frozen_params and p.requires_grad]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params and p.requires_grad]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if dist.get_rank == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def unfreeze_action_token_embeddings(model, lora_cfg):
        """
        Selectively unfreeze ONLY the action token rows in embed_tokens for LoRA training.

        All non-action token rows stay effectively frozen via a gradient mask hook —
        their gradients are zeroed before the optimizer update, so the language token
        embeddings are never modified.

        Since lm_head.weight is tied to embed_tokens.weight in Qwen3-VL (same tensor),
        this single hook also enables gradient flow through the action token logits in
        lm_head with no extra work.

        Requires:
          - lora_cfg.vlm_module_path: dot-path to the HF CausalLM inside `model`
            (e.g. "qwen_vl_interface.model")
          - The VLM interface parent (e.g. model.qwen_vl_interface) must expose
            _ACTION_TOKEN_MIN and _ACTION_TOKEN_MAX attributes.
        """
        vlm_module_path = lora_cfg.get("vlm_module_path", "qwen_vl_interface.model")
        attrs = vlm_module_path.split(".")

        # Navigate to the HF model to call get_input_embeddings()
        vlm_module = model
        for attr in attrs:
            vlm_module = getattr(vlm_module, attr)

        # Navigate to the VLM interface (parent of the HF model) for token range constants
        interface = model
        for attr in attrs[:-1]:
            interface = getattr(interface, attr)

        action_min = getattr(interface, "_ACTION_TOKEN_MIN", None)
        action_max = getattr(interface, "_ACTION_TOKEN_MAX", None)
        if action_min is None or action_max is None:
            raise ValueError(
                "Cannot find _ACTION_TOKEN_MIN / _ACTION_TOKEN_MAX on the VLM interface. "
                "Make sure the model is loaded from an -Action variant "
                "(produced by add_fast_tokens.sh)."
            )

        embed = vlm_module.get_input_embeddings()
        embed.weight.requires_grad_(True)

        def _action_only_grad_hook(grad):
            # Zero out every row except the action token range [action_min, action_max]
            masked = torch.zeros_like(grad)
            masked[action_min : action_max + 1] = grad[action_min : action_max + 1]
            return masked

        embed.weight.register_hook(_action_only_grad_hook)

        if dist.get_rank() == 0:
            n_action = action_max - action_min + 1
            print(
                f"Action-only embedding training enabled: "
                f"rows [{action_min}, {action_max}] ({n_action} tokens) will receive gradients; "
                f"all other embedding rows are masked to zero gradient."
            )
            print(f"  embed_tokens.weight.requires_grad = {embed.weight.requires_grad}  "
                  f"| shape = {tuple(embed.weight.shape)}")

            # Verify lm_head weight tying: in Qwen3-VL lm_head.weight should be the same
            # tensor as embed_tokens.weight.  If untied, gradients through lm_head action
            # logits would NOT flow back to embed_tokens, so action token output predictions
            # would not be learned.
            try:
                lm_head = vlm_module.base_model.model.lm_head
                is_tied = lm_head.weight is embed.weight
                print(
                    f"  lm_head.weight is embed_tokens.weight (tied) = {is_tied}  "
                    f"| lm_head.weight.requires_grad = {lm_head.weight.requires_grad}"
                )
                if not is_tied:
                    raise RuntimeError(
                        "lm_head.weight is NOT tied to embed_tokens.weight. "
                        "train_action_embeddings=True requires weight tying so that "
                        "action token output logits receive gradient updates. "
                        "Check the model's tie_word_embeddings config or use an -Action variant "
                        "produced by add_fast_tokens.sh."
                    )
            except AttributeError as e:
                raise RuntimeError(
                    "Cannot access vlm_module.base_model.model.lm_head to verify lm_head weight tying. "
                    "train_action_embeddings=True requires lm_head to be tied to embed_tokens "
                    "so that action token output logits receive gradient updates."
                ) from e

        return model

    @staticmethod
    def apply_lora_to_vlm(model, lora_cfg):
        """
        Apply PEFT LoRA adapters to the VLM backbone only.

        The path to the HuggingFace model is specified by lora_cfg.vlm_module_path
        (default: "qwen_vl_interface.model"). All other sub-modules (vision encoder,
        action_model, etc.) are unaffected; their params remain frozen by PEFT.

        Args:
            model: top-level nn.Module (e.g. QwenFast)
            lora_cfg: OmegaConf node with LoRA hyper-parameters:
                - r (int): LoRA rank
                - lora_alpha (int): LoRA scaling factor
                - lora_dropout (float): dropout on LoRA layers
                - target_modules (list[str]): projection names to apply LoRA
                - vlm_module_path (str): dot-path to the HF model inside `model`

        Returns:
            model with PEFT LoRA applied to the VLM backbone
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA training. Install with: pip install peft"
            ) from e

        vlm_module_path = lora_cfg.get("vlm_module_path", "qwen_vl_interface.model")
        attrs = vlm_module_path.split(".")

        # Navigate to the parent and the target HF model
        parent = model
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        vlm_model = getattr(parent, attrs[-1])

        target_modules = list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))

        lora_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            target_modules=target_modules,
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_vlm = get_peft_model(vlm_model, lora_config)

        if dist.get_rank() == 0:
            print("📐 LoRA applied to VLM backbone:")
            peft_vlm.print_trainable_parameters()

        setattr(parent, attrs[-1], peft_vlm)
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )

        # Log requires_grad status for action-related key parameters
        key_suffixes = ("embed_tokens.weight", "lm_head.weight")
        seen_data_ptrs = {}  # data_ptr -> first name, to detect tied tensors
        for name, param in model.named_parameters():
            if any(name.endswith(s) for s in key_suffixes):
                tied_note = ""
                if param.data_ptr() in seen_data_ptrs:
                    tied_note = f"  [tied to {seen_data_ptrs[param.data_ptr()]}]"
                else:
                    seen_data_ptrs[param.data_ptr()] = name
                print(f"  {name}: requires_grad={param.requires_grad}  shape={tuple(param.shape)}{tied_note}")

        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=False)
                if dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # 获取所有符合命名规则，支持 .pt 和 .safetensors
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) 
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # 确保是文件
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # 提取步数并排序
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # 按步数排序，获取最新的 checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")
