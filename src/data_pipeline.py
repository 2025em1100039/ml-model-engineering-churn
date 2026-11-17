from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop_duplicates(subset=["customerID"])
    y = df.pop("Churn").map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["customerID"])
    return X, y

def split_data(X, y, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
