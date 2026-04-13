import pandas as pd
from pathlib import Path

from app.predict import preprocess_input, predict_from_csv


def test_preprocess_input_removes_non_feature_columns():
    df = pd.DataFrame([
        {
            "binary_target": 1,
            "binary_label": "attack",
            "class_label": "DoS",
            "source_file": "file1",
            "direction": 1,
            "timestamp": 1234567890.0,
            "dlc": 8,
            "can_id_int": 100,
            "byte_0_int": 1,
            "byte_1_int": 2,
            "byte_2_int": 3,
            "byte_3_int": 4,
            "byte_4_int": 5,
            "byte_5_int": 6,
            "byte_6_int": 7,
            "byte_7_int": 8,
            "inter_arrival": 0.01,
            "payload_sum": 36,
            "nonzero_bytes": 8,
            "payload_unique_values": 8,
        }
    ])

    X = preprocess_input(df)

    assert "binary_target" not in X.columns
    assert "binary_label" not in X.columns
    assert "class_label" not in X.columns
    assert "source_file" not in X.columns
    assert "direction" not in X.columns
    assert "timestamp" not in X.columns
    assert "dlc" in X.columns
    assert "can_id_int" in X.columns


def test_preprocess_input_returns_numeric_dataframe():
    df = pd.DataFrame([
        {
            "dlc": 8,
            "can_id_int": 100,
            "byte_0_int": 1,
            "byte_1_int": 2,
            "byte_2_int": 3,
            "byte_3_int": 4,
            "byte_4_int": 5,
            "byte_5_int": 6,
            "byte_6_int": 7,
            "byte_7_int": 8,
            "inter_arrival": 0.01,
            "payload_sum": 36,
            "nonzero_bytes": 8,
            "payload_unique_values": 8,
        }
    ])

    X = preprocess_input(df)
    assert X.shape[0] == 1
    assert X.select_dtypes(include=["number"]).shape[1] == X.shape[1]


def test_predict_from_csv_creates_output_file(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame([
        {
            "dlc": 8,
            "can_id_int": 123,
            "byte_0_int": 10,
            "byte_1_int": 20,
            "byte_2_int": 0,
            "byte_3_int": 0,
            "byte_4_int": 5,
            "byte_5_int": 1,
            "byte_6_int": 0,
            "byte_7_int": 255,
            "inter_arrival": 0.002,
            "payload_sum": 291,
            "nonzero_bytes": 5,
            "payload_unique_values": 6,
        }
    ])
    df.to_csv(input_file, index=False)

    result_df = predict_from_csv(input_file, output_file)

    assert output_file.exists()
    assert "predicted_probability" in result_df.columns
    assert "predicted_class" in result_df.columns
    assert len(result_df) == 1