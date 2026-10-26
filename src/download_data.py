from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "heart.csv"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the UCI heart disease dataset...")

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    DATA_PATH.write_text(response.text, encoding="utf-8")

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?"
    )

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().copy()
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Rows available: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Target distribution:")
    print(df["target"].value_counts().to_string())
    return df


if __name__ == "__main__":
    download_data()
