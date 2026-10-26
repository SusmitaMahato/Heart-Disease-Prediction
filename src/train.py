from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "heartwise_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET = "target"


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python -m src.download_data"
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training the HeartWise model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation on held-out test set ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "target_name": TARGET
    }, MODEL_PATH)
    print(f"\nTrained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
