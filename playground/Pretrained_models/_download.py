from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Qwen/Qwen3-VL-4B-Instruct",
#     local_dir="./Qwen3-VL-4B-Instruct",
#     repo_type="model",
#     resume_download=True,
# )

snapshot_download(
    repo_id="Qwen/Qwen3.5-2B",
    local_dir="./Qwen3.5-2B",
    repo_type="model",
    resume_download=True,
)

'''
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
'''

snapshot_download(
    repo_id="physical-intelligence/fast",
    local_dir="./fast",
    repo_type="model",
    resume_download=True,
)
