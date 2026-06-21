import os
from huggingface_hub import create_repo, upload_folder
from src.config import SPACE_REPO_ID
HF_TOKEN=os.getenv("HF_TOKEN")
if not HF_TOKEN: raise EnvironmentError("HF_TOKEN is missing")
create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    token=HF_TOKEN,
    exist_ok=True,
    private=False
)
upload_folder(folder_path="deployment", repo_id=SPACE_REPO_ID, repo_type="space", token=HF_TOKEN)
print(f"Space deployed: https://huggingface.co/spaces/{SPACE_REPO_ID}")
