import os
from huggingface_hub import HfApi

repo_id = "ClickNoow/Terese-v11-adapter"

# Unified path directly to the adapter inside the checkpoint
local_folder_to_upload = os.path.join(
    "./models", 
    "Terese_v11", 
    "checkpoint-1398", 
    "adapter_model"
)

if not os.path.exists(local_folder_to_upload):
    print(f"Error: Folder not found at {local_folder_to_upload}")
    print("Please ensure the path is correct.")
else:
    print(f"Folder found: {local_folder_to_upload}")
    print(f"Starting upload to Hugging Face repo: {repo_id}...")

    api = HfApi()

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=local_folder_to_upload,
        repo_id=repo_id,
        repo_type="model"
    )

    print("\n--- Done! ---")
    print(f"Your adapter has been successfully uploaded to: https://huggingface.co/{repo_id}")