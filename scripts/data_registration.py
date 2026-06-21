import os
from huggingface_hub import create_repo, upload_file
from src.config import DATASET_REPO_ID
HF_TOKEN=os.getenv("HF_TOKEN")
if not HF_TOKEN: raise EnvironmentError("HF_TOKEN is missing")
create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN, exist_ok=True, private=False)
upload_file(
    path_or_fileobj="data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)
