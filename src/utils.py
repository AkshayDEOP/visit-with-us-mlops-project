import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df=df.drop(columns=[c for c in ["Unnamed: 0","CustomerID"] if c in df.columns], errors="ignore")
    if "Gender" in df.columns: df["Gender"]=df["Gender"].replace({"Fe Male":"Female"})
    if "MaritalStatus" in df.columns: df["MaritalStatus"]=df["MaritalStatus"].replace({"Single":"Unmarried"})
    return df.drop_duplicates()
