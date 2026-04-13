from pathlib import Path
import pandas as pd
from app.utils import (
    load_model,
    load_scaler,
    load_threshold,
    clean_input_dataframe,
    validate_input_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def load_artifacts():
    model = load_model()
    scaler = load_scaler()
    threshold = load_threshold()
    return model, scaler, threshold


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    validate_input_dataframe(df)
    return clean_input_dataframe(df)


def predict_from_csv(input_file, output_file):
    model, scaler, threshold = load_artifacts()

    df = pd.read_csv(input_file)
    X = preprocess_input(df)
    X_scaled = scaler.transform(X)

    probs = model.predict(X_scaled, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    df["predicted_probability"] = probs
    df["predicted_class"] = preds

    df.to_csv(output_file, index=False)
    return df