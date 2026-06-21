import os, pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, upload_file
from src.config import DATASET_REPO_ID,TARGET_COLUMN,RANDOM_STATE,TEST_SIZE
from src.utils import clean_data
HF_TOKEN=os.getenv("HF_TOKEN")
path=hf_hub_download(DATASET_REPO_ID,"tourism.csv",repo_type="dataset",token=HF_TOKEN)
df=clean_data(pd.read_csv(path))
X=df.drop(columns=[TARGET_COLUMN]); y=df[TARGET_COLUMN].astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
train=pd.concat([Xtr.reset_index(drop=True),ytr.reset_index(drop=True)],axis=1); test=pd.concat([Xte.reset_index(drop=True),yte.reset_index(drop=True)],axis=1)
os.makedirs("data",exist_ok=True); train.to_csv("data/train.csv",index=False); test.to_csv("data/test.csv",index=False)
upload_file(
    path_or_fileobj="data/train.csv",
    path_in_repo="train.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

upload_file(
    path_or_fileobj="data/test.csv",
    path_in_repo="test.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)
print("Prepared",train.shape,test.shape)
