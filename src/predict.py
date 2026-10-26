from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heartwise_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. First run: python -m src.train"
        )
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]


def predict_risk(input_data):
    model, features = load_model()
    input_df = pd.DataFrame([input_data], columns=features)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    positive_index = list(model.classes_).index(1)
    probability = float(probabilities[positive_index])

    return {
        "prediction": prediction,
        "probability": probability
    }
