# Binary CAN Intrusion Detection

This project implements a **binary intrusion detection system for automotive Controller Area Network (CAN) traffic** using a neural network baseline. The goal is to classify each CAN message as either **normal** or **attack** based on engineered message-level features extracted from cleaned CAN logs.

The repository now includes both the **training pipeline** and the **deployment/inference pipeline** for running the trained model locally in VS Code.

---

## Project Overview

Modern vehicles use the **Controller Area Network (CAN)** bus to allow communication between Electronic Control Units (ECUs). Although CAN is efficient and widely adopted, it lacks built-in authentication and encryption, making it vulnerable to attacks such as:

- message injection
- spoofing
- replay attacks
- denial-of-service (DoS)

To address this, this project applies machine learning to detect malicious CAN traffic. A tuned **multilayer perceptron (MLP)** was trained on a cleaned binary CAN dataset and evaluated using multiple classification metrics.

---

## Project Goals

The project was designed to:

- build a binary classifier for CAN intrusion detection
- preprocess and clean CAN traffic data
- engineer useful message-level features
- train and evaluate a neural network baseline
- tune a decision threshold for better class balance
- export the trained model and inference artifacts
- support local deployment and prediction in VS Code

---

## Features

- Binary classification of CAN traffic
- Clean preprocessing and feature engineering pipeline
- Handling of missing and infinite values
- Standard scaling of numeric features
- Neural network baseline using a tuned MLP
- Threshold tuning on the validation set
- Evaluation using:
  - accuracy
  - precision
  - recall
  - F1-score
  - ROC-AUC
  - balanced accuracy
  - specificity
- Export of:
  - trained models
  - scaler
  - threshold
  - test metrics
  - training history
  - plots
- Local deployment support for CSV-based prediction

---

## Dataset and Features

The model was trained on a cleaned binary CAN dataset:

- `can_binary_cleaned.csv`

### Target Variable

- `binary_target`
  - `0` = normal
  - `1` = attack

### Columns dropped before training

The following columns were removed during preprocessing because they were not used as model inputs or could introduce leakage/metadata dependencies:

- `binary_target`
- `binary_label`
- `class_label`
- `source_file`
- `direction`
- `timestamp`

### Input features used for training

The remaining numeric features were used for model training:

- `dlc`
- `can_id_int`
- `byte_0_int`
- `byte_1_int`
- `byte_2_int`
- `byte_3_int`
- `byte_4_int`
- `byte_5_int`
- `byte_6_int`
- `byte_7_int`
- `inter_arrival`
- `payload_sum`
- `nonzero_bytes`
- `payload_unique_values`

### Additional preprocessing steps

The pipeline includes:

- numeric type coercion
- replacement of infinite values with NaN
- median imputation for missing values
- fallback zero filling
- clipping extreme feature values to the range `[-1e6, 1e6]`
- `StandardScaler` normalization on the train/validation/test split

---

## Train / Validation / Test Split

The dataset was split as follows:

- **70% training**
- **15% validation**
- **15% test**

The split was stratified to preserve the class distribution.

---

## Model Architecture

The binary classifier is a tuned **multilayer perceptron (MLP)** built with TensorFlow/Keras.

### Architecture summary

- Input layer
- Dense(64, ReLU, L2 regularization)
- Batch Normalization
- Dropout(0.30)
- Dense(32, ReLU, L2 regularization)
- Batch Normalization
- Dropout(0.25)
- Dense(16, ReLU, L2 regularization)
- Dropout(0.20)
- Dense(1, Sigmoid)

### Training settings

- Optimizer: Adam
- Learning rate: `3e-5`
- Gradient clipping: `clipnorm=1.0`
- Loss: Binary Cross-Entropy
- Metrics:
  - accuracy
  - precision
  - recall
  - AUC

### Callbacks used

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- TerminateOnNaN

---

## Threshold Tuning

Instead of using the default threshold of `0.50`, the model tunes the classification threshold on the **validation set**.

Thresholds from `0.10` to `0.90` were evaluated, and the final threshold was selected using:

1. **balanced accuracy**
2. **F1-score** as a tie-breaker

The selected threshold was saved for deployment and later inference.

---

## Final Results

The trained binary classifier achieved strong performance on the held-out test set.

### Reported test metrics

- **Accuracy:** 0.9800
- **Precision:** 0.9835
- **Recall:** 0.9955
- **F1-score:** 0.9894
- **ROC-AUC:** 0.9627
- **Balanced Accuracy:** 0.8579
- **Specificity:** 0.7203

These results show that the model is very strong at detecting attacks, with very high recall and F1-score, while still maintaining solid discrimination ability overall.

---

## Saved Artifacts

The training pipeline exports the following files:

### Models
- `best_mlp_binary_model.keras`
- `final_mlp_binary_model.keras`

### Preprocessing artifacts
- `binary_scaler.pkl`

### Saved evaluation data
- `binary_test_data.npz`

### Metrics and logs
- `mlp_binary_test_metrics.json`
- `best_threshold_binary.json`
- `threshold_tuning_binary.csv`
- `training_history_binary.csv`

### Plots
- `loss_curve_binary.png`
- `accuracy_curve_binary.png`
- `precision_curve_binary.png`
- `recall_curve_binary.png`
- `auc_curve_binary.png`
- `roc_curve_binary.png`
- `precision_recall_curve_binary.png`
- `confusion_matrix_binary.png`

---

## Local Deployment

The repository now supports **local inference in VS Code** using the exported deployment files.

### Deployment files used

- trained model: `.keras`
- scaler: `.pkl`
- threshold file: `.json`

### Inference workflow

1. Load trained model
2. Load saved scaler
3. Load tuned threshold
4. Read input CSV
5. Drop unused metadata columns
6. Keep numeric features
7. Fill missing values
8. Clip extreme values
9. Scale features
10. Predict attack probability
11. Convert probability to class label using tuned threshold

---

## Current Repository Structure

```text
can-ids-binary-detector/
├── app/
│   ├── __init__.py
│   ├── predict.py
│   └── utils.py
├── logs/
│   └── best_threshold_binary.json
├── models/
│   ├── best_mlp_binary_model.keras
│   └── binary_scaler.pkl
├── notebooks/
│   ├── demo_inference.ipynb
│   └── README.md
├── sample_data/
│   └── sample_input.csv
├── tests/
│   ├── __init__.py
│   └── test_predict.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
