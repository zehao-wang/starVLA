"""
Test script: inspect how FAST action tokens are tokenized by the Qwen3.5-Action VLM tokenizer.

Questions answered:
  1. Is the number of <robot_action_N> tokens fixed or variable across samples?
  2. Is the '|' terminator tokenized as a single, correct token?
  3. What do the full token-ID sequences look like?

Usage (from repo root):
    python scripts/tests/test_fast_tokenization.py

Requirements: the model must be present at playground/Pretrained_models/Qwen3.5-2B-Action
"""

import sys
import os
import numpy as np

# --- repo root on path ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Config (mirrors run_robotwin_train_qwen35_fast_delta50.sh)
# ---------------------------------------------------------------------------
FAST_TOKENIZER_PATH = os.path.join(REPO_ROOT, "playground/Pretrained_models/fast")
ACTION_MODEL_PATH   = os.path.join(REPO_ROOT, "playground/Pretrained_models/Qwen3.5-2B-Action")
TIME_HORIZON  = 50   # future_action_window_size 49 + 1
ACTION_DIM    = 14   # delta_qpos bimanual

# ---------------------------------------------------------------------------
# 1. Load FAST BPE tokenizer (action → token-id list)
# ---------------------------------------------------------------------------
from starVLA.model.modules.action_model.fast_ActionHeader import _load_fast_processor

print("=" * 60)
print("Loading FAST action tokenizer from:", FAST_TOKENIZER_PATH)
fast_proc = _load_fast_processor(FAST_TOKENIZER_PATH)
fast_proc.time_horizon = TIME_HORIZON
fast_proc.action_dim   = ACTION_DIM

# ---------------------------------------------------------------------------
# 2. Load Qwen3.5-Action VLM tokenizer
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer

print("Loading Qwen3.5-Action VLM tokenizer from:", ACTION_MODEL_PATH)
vlm_tokenizer = AutoTokenizer.from_pretrained(ACTION_MODEL_PATH)

ACTION_TOKEN_MIN = 248077           # from added_custom_token_id_map.json (Qwen3.5-2B-Action)
ACTION_TOKEN_MAX = 248077 + 2047    # 2048 action tokens (250124)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def fast_tokens_to_vlm_string(fast_token_ids) -> str:
    """Mirrors Qwen35Fast.map_fast_token_to_vlm_action."""
    body = "".join(f"<robot_action_{t}>" for t in fast_token_ids)
    return f"{body}|"


# ---------------------------------------------------------------------------
# 3. Encode a batch of random actions and inspect token counts
# ---------------------------------------------------------------------------
np.random.seed(42)
N_SAMPLES = 5
raw_actions = [np.random.uniform(-1, 1, (TIME_HORIZON, ACTION_DIM)).astype(np.float32)
               for _ in range(N_SAMPLES)]

batch = np.stack(raw_actions, axis=0)  # (N, T, D)
batch_fast_tokens = fast_proc(batch)   # List[List[int]]  (one per sample)

print("\n" + "=" * 60)
print(f"FAST BPE token counts per sample  (time_horizon={TIME_HORIZON}, action_dim={ACTION_DIM})")
print(f"  max possible chars = {TIME_HORIZON * ACTION_DIM} = {TIME_HORIZON} x {ACTION_DIM}")
print("-" * 60)

for i, tokens in enumerate(batch_fast_tokens):
    bpe_char_count = len(fast_proc.bpe_tokenizer.decode(tokens))
    print(f"  sample {i}: fast_token_count={len(tokens):>4d}  "
          f"decoded_chars={bpe_char_count:>4d}")

all_lens = [len(t) for t in batch_fast_tokens]
print(f"\n  Min / Max fast-token count: {min(all_lens)} / {max(all_lens)}")
if min(all_lens) == max(all_lens):
    print("  => FIXED length (BPE produced same token count for all samples)")
else:
    print("  => VARIABLE length (BPE merge count differs across samples)")


# ---------------------------------------------------------------------------
# 4. Inspect VLM tokenization of the action string for sample 0
# ---------------------------------------------------------------------------
sample_fast_tokens = batch_fast_tokens[0]
vlm_string = fast_tokens_to_vlm_string(sample_fast_tokens)

print("\n" + "=" * 60)
print("VLM action string for sample 0  (first 3 tokens shown):")
# Show only first few special tokens to keep output readable
preview_tokens = sample_fast_tokens[:3]
preview_str = fast_tokens_to_vlm_string(preview_tokens)
print(f"  Preview string: {preview_str!r}")

# Tokenize the preview
preview_ids = vlm_tokenizer.encode(preview_str, add_special_tokens=False)
print(f"  Token IDs:      {preview_ids}")
print(f"  Decoded back:   {[vlm_tokenizer.decode([tid]) for tid in preview_ids]}")

# Check each token in preview
print("\n  Token-by-token breakdown:")
for raw_tok, vlm_id in zip(
        [f"<robot_action_{t}>" for t in preview_tokens] + ["|"],
        preview_ids[:len(preview_tokens) + 1]):
    is_action = ACTION_TOKEN_MIN <= vlm_id <= ACTION_TOKEN_MAX
    decoded = vlm_tokenizer.decode([vlm_id])
    print(f"    {raw_tok!r:30s}  id={vlm_id:7d}  decoded={decoded!r}  action_token={is_action}")


# ---------------------------------------------------------------------------
# 5. Tokenize the full action string for sample 0 and report stats
# ---------------------------------------------------------------------------
full_ids = vlm_tokenizer.encode(vlm_string, add_special_tokens=False)
action_ids   = [i for i in full_ids if ACTION_TOKEN_MIN <= i <= ACTION_TOKEN_MAX]
pipe_ids     = [i for i in full_ids if vlm_tokenizer.decode([i]) == "|"]
other_ids    = [i for i in full_ids if i not in action_ids and i not in pipe_ids]

print("\n" + "=" * 60)
print("Full VLM tokenization of action string (sample 0):")
print(f"  fast token count        : {len(sample_fast_tokens)}")
print(f"  VLM token count (total) : {len(full_ids)}")
print(f"    - action special tokens: {len(action_ids)}")
print(f"    - '|' tokens           : {len(pipe_ids)}")
print(f"    - other tokens         : {len(other_ids)}")

if len(action_ids) == len(sample_fast_tokens):
    print("\n  => Each <robot_action_N> maps to exactly ONE VLM token. [CORRECT]")
else:
    print(f"\n  => MISMATCH: {len(sample_fast_tokens)} fast tokens but {len(action_ids)} action VLM tokens! [PROBLEM]")

if len(pipe_ids) == 1:
    print(f"  => '|' is a single token (id={pipe_ids[0]}). [CORRECT]")
elif len(pipe_ids) == 0:
    print(f"  => '|' was NOT found as a standalone token — may be merged into an adjacent token. [CHECK]")
    tail = vlm_tokenizer.decode(full_ids[-3:])
    print(f"     Last 3 decoded tokens: {[vlm_tokenizer.decode([i]) for i in full_ids[-3:]]}")
else:
    print(f"  => Multiple '|' tokens found: {pipe_ids}. [UNEXPECTED]")

if other_ids:
    print(f"\n  => Unexpected non-action tokens in action string: {other_ids}")
    print(f"     Decoded: {[vlm_tokenizer.decode([i]) for i in other_ids]}")
else:
    print(f"  => No stray non-action tokens. [CLEAN]")


# ---------------------------------------------------------------------------
# 6. Confirm offset mapping: fast_token_id 0 => vlm id ACTION_TOKEN_MIN
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Offset sanity check:")
tok0_id = vlm_tokenizer.encode("<robot_action_0>", add_special_tokens=False)
match0 = tok0_id == [ACTION_TOKEN_MIN]
print(f"  <robot_action_0>    => VLM id {tok0_id}  (expected {[ACTION_TOKEN_MIN]})  {'[OK]' if match0 else '[MISMATCH]'}")
tok_last_id = vlm_tokenizer.encode("<robot_action_2047>", add_special_tokens=False)
match_last = tok_last_id == [ACTION_TOKEN_MAX]
print(f"  <robot_action_2047> => VLM id {tok_last_id}  (expected {[ACTION_TOKEN_MAX]})  {'[OK]' if match_last else '[MISMATCH]'}")

print("\nDone.")
