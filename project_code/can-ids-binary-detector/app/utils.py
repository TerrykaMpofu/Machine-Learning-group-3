from pathlib import Path
import json
import pickle
import pandas as pd
import tensorflow as tf

DROP_COLS = [
    "binary_target",
    "binary_label",
    "class_label",
    "source_file",
    "direction",
    "timestamp",
]


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def get_paths():
    base_dir = get_base_dir()

    return {
        "base_dir": base_dir,
        "model_path": base_dir / "models" / "best_mlp_binary_model.keras",
        "scaler_path": base_dir / "models" / "binary_scaler.pkl",
        "threshold_path": base_dir / "logs" / "best_threshold_binary.json",
    }


def load_model():
    paths = get_paths()
    return tf.keras.models.load_model(paths["model_path"])


def load_scaler():
    paths = get_paths()
    with open(paths["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    return scaler


def load_threshold(default: float = 0.5) -> float:
    paths = get_paths()
    threshold_path = paths["threshold_path"]

    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            data = json.load(f)
        return float(data.get("best_threshold", default))

    return default


def clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.clip(lower=-1e6, upper=1e6)
    return X


def validate_input_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    X = df.drop(columns=DROP_COLS, errors="ignore")
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric feature columns found in input data.")