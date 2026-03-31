# vc_Robotwin2 — starVLA on RoboTwin2

Example configs and scripts for training and evaluating starVLA on the
[RoboTwin2](https://github.com/TianxingChen/RoboTwin) benchmark.

## Directory layout

```
vc_Robotwin2/
├── train_files/            Training configs and launch scripts
│   ├── starvla_cotrain_robotwin_qwen3vl.yaml
│   ├── starvla_cotrain_robotwin_qwen3vl_lora.yaml
│   ├── modality.json
│   ├── add_fast_tokens.sh
│   ├── run_robotwin_train_qwen3vl_fast.sh
│   └── run_robotwin_train_qwen3vl_fast_lora.sh
│
└── eval_files/             Evaluation scripts and helpers
    ├── deploy_policy.yml           Shared config (ckpt, port, action mode …)
    ├── model2robotwin_interface.py Policy client (WebSocket → server)
    │
    ├── run_eval_distributed.sh     ★ Our eval — multi-GPU, dataset-mode (recommended)
    ├── run_policy_server_gpu.sh    Start one policy server on one GPU
    ├── run_policy_server.sh        Start one policy server (manual / debug)
    ├── eval.sh                     Official RoboTwin eval — random-scene rollout
    ├── eval_client.py              Single-GPU dataset eval Python entry point
    │
    └── utils/
        ├── robotwin_helpers.py     Shared helpers (dataset I/O, env build, eval loop)
        ├── eval_orchestrator.py    Orchestrator: queue + worker management
        └── eval_worker.py          Worker process: one per GPU service
```

---

## Prerequisites

| Conda env | Purpose |
|-----------|---------|
| `starVLA` | Policy server (model inference, GPU) |
| `RoboTwin` | Simulation / eval client (CPU-heavy) |

Run all top-level shell scripts from `packages/starVLA/` with the `starVLA`
env active.  The scripts themselves switch to `RoboTwin` via `conda run` where
needed.

### Environment variables (optional overrides)

```bash
export DEPLOY_POLICY_YML=/path/to/my_config.yml   # default: eval_files/deploy_policy.yml
export HF_LEROBOT_HOME=/path/to/hf_lerobot        # default: .../data/robotwin2/hf_lerobot
export ROBOTWIN_CONDA_ENV=RoboTwin                 # default: RoboTwin
export STAR_VLA_PYTHON=/path/to/python             # default: ~/.conda/envs/starVLA/bin/python
export ROBOTWIN_PATH=/path/to/packages/RoboTwin    # auto-derived if unset
```

---

## Evaluation

There are two distinct evaluation pipelines:

| | Official (random rollout) | Ours (dataset-mode) |
|---|---|---|
| Scene initialisation | Random seed, random object poses | Fixed seed + initial qpos from dataset |
| Reproducibility | No | Yes — same scenes as data collection |
| Entry point | `eval.sh` | `run_eval_distributed.sh` |
| GPU utilisation | Single GPU | One server per GPU, full node |
| Result path | per RoboTwin convention | `results/robotwin2_dataset/<exp_name>/` |

---

### Official eval — random-scene rollout (`eval.sh`)

Uses RoboTwin's native `script/eval_policy.py`.  Each episode is initialised
with a fresh random seed, matching the official benchmark protocol.

```bash
# Run from packages/starVLA/

bash examples/vc_Robotwin2/eval_files/eval.sh \
    <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]
```

Example:

```bash
bash examples/vc_Robotwin2/eval_files/eval.sh \
    block_hammer_beat  block_hammer_beat_D0  starvla_demo  0  0
```

---

### Our eval — dataset-mode, multi-GPU distributed (`run_eval_distributed.sh`)

Replays episodes from a LeRobot dataset with fixed seeds and initial joint
positions, enabling reproducible evaluation on the exact scenes seen during
data collection.

Starts **one policy server per GPU**, then runs a pool of eval workers that
share a task queue.  Every GPU stays busy for the full duration.

```bash
# Run from packages/starVLA/

# Auto-detect all GPUs on the node
bash examples/vc_Robotwin2/eval_files/run_eval_distributed.sh \
    <hf_dataset_name>

# Full argument form
bash examples/vc_Robotwin2/eval_files/run_eval_distributed.sh \
    <hf_dataset_name>   \   # e.g. lerobot_robotwin_rand20k
    <exp_name>          \   # default: starvla_dist_eval
    <max_episodes>      \   # default: none  (all episodes per task)
    <gpu_ids>           \   # default: auto  (e.g. "0,1,2,3")
    <base_port>             # default: 5694  (ports = base, base+1, …)
```

Example — 4 GPUs, limit to 20 episodes per task:

```bash
bash examples/vc_Robotwin2/eval_files/run_eval_distributed.sh \
    lerobot_robotwin_rand20k  my_exp  20  "0,1,2,3"
```

#### Single-GPU debug

```bash
# 1. Start server
bash examples/vc_Robotwin2/eval_files/run_policy_server.sh

# 2. Run client directly (RoboTwin env, PYTHONPATH set manually)
export PYTHONPATH=packages/RoboTwin:packages/starVLA:examples/vc_Robotwin2/eval_files:$PYTHONPATH
conda run -n RoboTwin python examples/vc_Robotwin2/eval_files/eval_client.py \
    --repo-id  <hf_dataset_name>/Randomized \
    --task-name <task_name> \
    --exp-name  debug \
    --policy-config examples/vc_Robotwin2/eval_files/deploy_policy.yml
```

---

## Configuration — `deploy_policy.yml`

All client and server parameters live in `eval_files/deploy_policy.yml`.

```yaml
# Policy server connection
host: "127.0.0.1"
port: 5694        # single-GPU default
base_port: 5694   # distributed eval: ports = base_port, base_port+1, …

# Checkpoint
policy_ckpt_path: "results/Checkpoints/<run>/final_model/pytorch_model.pt"
unnorm_key: "new_embodiment"

# Action decoding
action_mode: "abs"            # abs | delta | rel
normalization_mode: "min_max" # min_max | q99

# Receding-horizon execution (action jitter reduction)
# null  → execute the full predicted chunk before re-predicting (default)
# int   → re-predict every exec_horizon steps; only the first exec_horizon
#         actions of each chunk are used.  Typical: chunk_size // 2 (e.g. 8)
exec_horizon: null
```

---

## Design decisions (our eval)

### Why one server per GPU?

The policy server is the GPU bottleneck; the RoboTwin simulation is CPU-bound.
Mapping one server to one GPU keeps every card saturated and lets the eval
workers run as many parallel simulations as needed without fighting over a
single inference endpoint.

### Task-level queue granularity

The task queue holds **one item per task** (all episodes for that task bundled
together), not one item per episode.  This avoids tearing down and
re-initialising the RoboTwin physics environment between episodes of the same
task, which is expensive.  Workers pull the next task only after finishing all
episodes of the current one.

### `spawn` multiprocessing start method

`multiprocessing.set_start_method("spawn")` is used instead of the default
`fork`.  RoboTwin relies on the SAPIEN physics engine (a C++ extension); CUDA
contexts and SAPIEN internal state are unsafe to inherit across a `fork`.
`spawn` gives each worker a clean process image.

### Receding-horizon execution (`exec_horizon`)

With a 16-step action chunk the model predicts all 16 future actions from a
single observation.  When the chunk is exhausted the next prediction starts
from a fresh observation, but there is no continuity constraint between the
last action of chunk *N* and the first action of chunk *N+1* — this causes a
visible joint-position jump at the boundary.

Setting `exec_horizon < action_chunk_size` (e.g. 8) re-predicts with fresh
observations halfway through each chunk:

```
chunk_size = 16,  exec_horizon = 8

step  0: predict from obs₀ → actions[0..15];  execute actions[0]
step  1: (reuse)                               execute actions[1]
...
step  7: (reuse)                               execute actions[7]
step  8: predict from obs₈ → actions[0..15];  execute actions[0]   ← smoother transition
...
```

The trade-off is inference frequency: 2× more server calls per episode.
For flow-matching models with `num_ddim_steps > 1` this is non-trivial; start
with `exec_horizon: 8` and tune from there.

### PYTHONPATH is set once in the shell layer

`ROBOTWIN_PATH`, `STARVLA_PATH`, and `EVAL_FILES_PATH` are added to
`PYTHONPATH` in `run_eval_distributed.sh` (and `eval.sh` for the official
eval).  Python code never hard-codes these paths.
