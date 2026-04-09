# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Qwen3.5-specific version: uses Qwen3_5ForConditionalGeneration instead of Qwen3VLForConditionalGeneration.

import argparse
import json
import os
from typing import Dict, Tuple

import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Qwen3_5ForConditionalGeneration not found. Please install transformers >= 5.2.0."
    ) from e

def _get_initializer_range(model) -> float:
    """
    Try best effort to get initializer std from config.
    """
    cfg = getattr(model.config, "text_config", model.config)
    std = getattr(cfg, "initializer_range", None)
    if std is None:
        std = getattr(model.config, "initializer_range", None)
    return float(std) if std is not None else 0.02

@torch.no_grad()
def _init_added_token_embeddings(
    model,
    old_vocab_size: int,
    num_added: int,
    init_mode: str = "mean+noise",
    noise_std: float | None = None,
    seed: int = 42,
):
    """
    Initialize newly-added token embeddings after resize_token_embeddings.

    Args:
        model: HF model
        old_vocab_size: vocab size BEFORE adding tokens
        num_added: number of tokens newly added
        init_mode:
            - "mean+noise": new = mean(old_emb) + N(0, noise_std)
            - "sample+noise": new = old_emb[random_row] + N(0, noise_std)
            - "hf_default": do nothing (keep HF default init)
        noise_std: if None, use model initializer_range (usually ~0.02)
        seed: for reproducibility
    """
    if num_added <= 0:
        return

    in_emb = model.get_input_embeddings()
    if in_emb is None or not hasattr(in_emb, "weight"):
        raise RuntimeError("Model has no input embeddings or unexpected embedding module.")

    w_in = in_emb.weight  # [new_vocab, hidden]
    device = w_in.device
    dtype = w_in.dtype
    hidden = w_in.shape[1]

    if noise_std is None:
        noise_std = _get_initializer_range(model)

    if init_mode == "hf_default":
        print("[Init] Using HF default init for newly added tokens.")
        return

    # Make randomness reproducible
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Stats computed only from old rows (before tokens are added)
    old_rows = w_in[:old_vocab_size]  # [old_vocab, hidden]

    if init_mode == "mean+noise":
        # mean vector in float32 (small tensor)
        mean = old_rows.mean(dim=0, keepdim=True).float()  # [1, hidden]
        noise = torch.randn((num_added, hidden), device=device, dtype=torch.float32, generator=g) * float(noise_std)
        new_rows = (mean + noise).to(dtype)

    elif init_mode == "sample+noise":
        # sample existing embeddings to keep diversity
        idx = torch.randint(low=0, high=old_vocab_size, size=(num_added,), device=device, generator=g)
        base = old_rows.index_select(0, idx).float()  # [num_added, hidden]
        noise = torch.randn((num_added, hidden), device=device, dtype=torch.float32, generator=g) * float(noise_std)
        new_rows = (base + noise).to(dtype)

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}. Use 'mean+noise', 'sample+noise', or 'hf_default'.")

    start = old_vocab_size
    end = old_vocab_size + num_added
    print(f"[Init] Writing new embeddings to rows [{start}, {end}) with mode={init_mode}, noise_std={noise_std}")
    w_in.data[start:end].copy_(new_rows)

    # If output embeddings exist and are NOT tied, also update them
    out_emb = model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
        w_out = out_emb.weight
        try:
            tied = (w_out.data_ptr() == w_in.data_ptr())
        except Exception:
            tied = False

        if (w_out.shape[0] == w_in.shape[0]) and (not tied):
            w_out.data[start:end].copy_(new_rows.to(w_out.dtype))
            print("[Init] Also initialized output embeddings (not tied).")
        else:
            print("[Init] Output embeddings tied or shape mismatch; skip explicit init.")



def add_new_tokens(
    model,
    processor,
    token_len: int = 32,
    init_mode: str = "mean+noise",
    noise_std: float | None = None,
    seed: int = 42,
    save_dir=None
) -> Tuple[Dict[str, int], int, int, int]:
    """
    Inject <|action_i|> tokens into Qwen tokenizer, resize model, initialize embeddings, save.
    """
    tokenizer = processor.tokenizer
    old_vocab_size = len(tokenizer)

    base_tok_size = tokenizer.vocab_size
    model_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"[DEBUG] tokenizer.vocab_size(base) = {base_tok_size}")
    print(f"[DEBUG] len(tokenizer)(total)      = {old_vocab_size}")
    print(f"[DEBUG] model.embed_size(before)   = {model_embed_size}")
    print(f"[DEBUG] added_in_tokenizer         = {old_vocab_size - base_tok_size}")

    action_tokens = [f"<|action_{i}|>" for i in range(token_len)]
    print("Add action tokens:", action_tokens)

    num_added = tokenizer.add_tokens(action_tokens, special_tokens=False)
    print("Added action tokens to Qwen:", num_added)
    print("New Qwen vocab size:", len(tokenizer))

    # Resize embeddings (required)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

        # Initialize the newly-added rows in embedding matrix
        _init_added_token_embeddings(
            model=model,
            old_vocab_size=old_vocab_size,
            num_added=num_added,
            init_mode=init_mode,
            noise_std=noise_std,
            seed=seed,
        )
    
    # Save
    if save_dir is None:
        raise
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print("Qwen + Action Query vocab saved to:", save_dir)

def unit_test_qwen_encode_action_span(qwen_tokenizer, token_len=32):
    """
    Test Qwen tokenizer can encode the action-span string into expected tokens.
    """
    # You can also try " ".join(...) if you want strict boundary.
    text = "".join([f"<|action_{i}|>" for i in range(token_len)])

    ids = qwen_tokenizer.encode(text, add_special_tokens=False)
    toks = qwen_tokenizer.convert_ids_to_tokens(ids)
    print("[Qwen encode test] text:", repr(text))
    print("[Qwen encode test] len(ids):", len(ids))
    print("[Qwen encode test] ids:", ids)
    print("[Qwen encode test] toks:", toks)

    assert len(ids) == token_len, f"Expected {token_len} tokens, got {len(ids)}"
    for i in range(token_len):
        expected_tok = f"<|action_{i}|>"
        assert toks[i] == expected_tok, f"Token {i} mismatch: expected {expected_tok}, got {toks[i]}"


def main():
    parser = argparse.ArgumentParser(
        description="Add special tokens to Qwen3.5 model and save to local directory."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B", help="HF Hub model ID or local path")
    parser.add_argument("--save-dir", required=True, help="Output directory to save")
    parser.add_argument("--init-strategy", default="mean+noise", choices=["sample+noise", "mean+noise"], help="Initialization strategy for newly added embeddings")
    parser.add_argument("--token-len", default=64, type=int)
    args = parser.parse_args()


    print(f"[INFO] Loading Qwen3.5 model: {args.model_id}")

    qwen_processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )


    add_new_tokens(
        model=qwen_model,
        processor=qwen_processor,
        token_len=args.token_len,
        init_mode=args.init_strategy,
        noise_std=None,           # or explicitly set e.g. 0.02 / 0.01
        seed=42,
        save_dir=args.save_dir
    )

     # Reload tokenizer for unit test
    qwen_processor = AutoProcessor.from_pretrained(args.save_dir, trust_remote_code=True)
    qwen_tokenizer = qwen_processor.tokenizer
    unit_test_qwen_encode_action_span(qwen_tokenizer, token_len=args.token_len)



if __name__ == "__main__":
    main()
