# train_binary_model.py
# ============================================================
# Binary CAN intrusion detection training script
# ------------------------------------------------------------
# This script:
# 1. Loads the cleaned binary CAN dataset
# 2. Removes unnecessary / leakage columns
# 3. Cleans NaN and infinite values
# 4. Splits the data into train and validation sets
# 5. Scales the numeric features
# 6. Builds and trains a simple MLP neural network
# 7. Evaluates performance on the validation set
# 8. Saves the trained model, metrics, and training curves
# ============================================================

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn tools for preprocessing, splitting, weighting, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# TensorFlow / Keras tools for building and training the neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. Basic settings

# Configure pandas so more columns are visible when printing
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Set a default plot size for saved figures
plt.rcParams["figure.figsize"] = (10, 5)

# Fix random seeds for reproducibility
# This helps ensure similar train/validation splits and model initialization
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define the main project folders
BASE_DIR = Path.home() / "projects" / "can_ids"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "binary"
LOG_DIR = BASE_DIR / "logs" / "binary"

# Create folders if they do not already exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Full path to the cleaned binary dataset
BINARY_FILE = DATA_DIR / "can_binary_cleaned.csv"

# Print run information so it is easy to verify paths in HPC logs
print("=" * 80)
print("CAN IDS Binary Neural Network Training")
print("=" * 80)
print(f"Base directory      : {BASE_DIR}")
print(f"Data directory      : {DATA_DIR}")
print(f"Binary dataset path : {BINARY_FILE}")
print()



# 2. Check file exists

# Stop immediately if the dataset file cannot be found
if not BINARY_FILE.exists():
    raise FileNotFoundError(
        f"Binary dataset not found: {BINARY_FILE}\n"
        "Make sure your dataset is in ~/projects/can_ids/data/"
    )


# 3. Load dataset

# Read the binary classification dataset into a pandas DataFrame
binary_df = pd.read_csv(BINARY_FILE)

# Print basic information so we can confirm the file loaded correctly
print("Loaded binary dataset successfully.")
print(f"Binary dataset shape: {binary_df.shape}")
print()

print("Binary columns:")
print(binary_df.columns.tolist())
print()

print("Binary dataset info:")
binary_df.info()
print()



# 4. Prepare binary dataset

# Work on a copy of the original DataFrame so the original stays unchanged
df = binary_df.copy()

# Make sure the expected binary target column exists
if "binary_target" not in df.columns:
    raise ValueError("Expected column 'binary_target' not found in binary dataset.")

# Extract the target column
# This is what the model will learn to predict:
# 0 = normal, 1 = attack
y = df["binary_target"].copy()

# These columns are removed from the input features:
# - binary_target: target column
# - binary_label: text version of the target
# - class_label: multiclass label, leaks label information
# - source_file: metadata, not a predictive feature
# - direction: often text / metadata
# - timestamp: dropped here for a simpler and more stable baseline
drop_cols = [
    "binary_target",
    "binary_label",
    "class_label",
    "source_file",
    "direction",
    "timestamp"
]

# Keep the remaining columns as candidate features
X = df.drop(columns=drop_cols, errors="ignore")

# Keep only numeric columns because the neural network expects numeric input
X = X.select_dtypes(include=[np.number]).copy()

# Stop if no usable feature columns remain
if X.empty:
    raise ValueError("No numeric feature columns found after preprocessing.")

# Print the initial feature setup
print("Initial feature columns used for binary training:")
print(X.columns.tolist())
print()
print(f"Initial X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Target distribution:")
print(y.value_counts(dropna=False))
print()


# Clean target column

# Remove rows where the target is missing
valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()

# Convert the target values to integers
y = y.astype(int)

# Keep only rows where the target is truly binary (0 or 1)
# This protects against accidental bad labels
binary_mask = y.isin([0, 1])
X = X.loc[binary_mask].copy()
y = y.loc[binary_mask].copy()

print("Target distribution after target cleanup:")
print(y.value_counts(dropna=False))
print()


# Inspect NaN and inf before cleaning

# Show how many missing values each feature has before cleanup
print("Missing values per feature before cleaning:")
print(X.isna().sum().sort_values(ascending=False))
print()

# Convert to NumPy for easy inspection of inf / NaN values
numeric_array = X.to_numpy(dtype=float)
print("Total inf values before cleaning:", np.isinf(numeric_array).sum())
print("Total NaN values before cleaning:", np.isnan(numeric_array).sum())
print()


# Clean feature matrix

# Replace positive/negative infinity with NaN first
# This allows them to be filled in the next step
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing numeric values with each column's median
# Median is a robust choice for tabular data
feature_medians = X.median(numeric_only=True)
X = X.fillna(feature_medians)

# If any column still has missing values after median fill
# (for example, if the whole column was NaN), fill the rest with 0
X = X.fillna(0)

# Clip extremely large values to reduce the chance of unstable training
X = X.clip(lower=-1e12, upper=1e12)

# Confirm the cleaning worked
print("Missing values per feature after cleaning:")
print(X.isna().sum().sort_values(ascending=False))
print()

clean_array = X.to_numpy(dtype=float)
print("Total inf values after cleaning:", np.isinf(clean_array).sum())
print("Total NaN values after cleaning:", np.isnan(clean_array).sum())
print()

print("Final feature columns used for training:")
print(X.columns.tolist())
print()
print(f"Final X shape: {X.shape}")
print(f"Final y shape: {y.shape}")
print()



# 5. Train / validation split

# Split the dataset into:
# - training set (85%)
# - validation set (15%)
# We stratify by y so the class balance stays similar in both sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=SEED
)

# Optional train/val/test split for later use
# Uncomment this block if you want a separate held-out test set
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X,
#     y,
#     test_size=0.30,
#     stratify=y,
#     random_state=SEED
# )
#
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp,
#     y_temp,
#     test_size=0.50,
#     stratify=y_temp,
#     random_state=SEED
# )

print("Data split complete:")
print(f"Train      : {X_train.shape}, {y_train.shape}")
print(f"Validation : {X_val.shape}, {y_val.shape}")
# print(f"Test       : {X_test.shape}, {y_test.shape}")
print()



# 6. Scale features

# Standardize features so they have similar scale
# Fit ONLY on training data, then apply the same transformation to validation
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)

print("Scaling complete:")
print(f"Scaled train shape: {X_train_scaled.shape}")
print(f"Scaled val shape  : {X_val_scaled.shape}")
# print(f"Scaled test shape : {X_test_scaled.shape}")
print()

# Sanity-check the scaled data to make sure no bad values remain
print("Post-scaling sanity checks:")
print("NaN in X_train_scaled:", np.isnan(X_train_scaled).any())
print("NaN in X_val_scaled  :", np.isnan(X_val_scaled).any())
print("Inf in X_train_scaled:", np.isinf(X_train_scaled).any())
print("Inf in X_val_scaled  :", np.isinf(X_val_scaled).any())
print("y_train unique values:", np.unique(y_train))
print("y_val unique values  :", np.unique(y_val))
print()

# Stop immediately if the scaled data is still invalid
if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
    raise ValueError("NaN values detected after scaling. Stop and inspect preprocessing.")

if np.isinf(X_train_scaled).any() or np.isinf(X_val_scaled).any():
    raise ValueError("Inf values detected after scaling. Stop and inspect preprocessing.")



# 7. Compute class weights

# If the dataset is imbalanced, class weights tell the model
# to pay more attention to the minority class during training
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

print("Class weights:")
print(class_weight)
print()



# 8. Build binary MLP model

# Create a simple feedforward neural network (MLP)
# - Input layer matches number of features
# - Hidden layers learn nonlinear patterns
# - Dropout helps reduce overfitting
# - Final sigmoid layer outputs probability of "attack"
mlp_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model:
# - Adam optimizer for gradient descent
# - binary_crossentropy because this is a binary classification task
# - learning rate lowered for stability
# - extra metrics help monitor security-relevant performance
mlp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("Binary model summary:")
mlp_model.summary()
print()

# 9. Callbacks

# Define where the best model should be saved during training
best_model_path = MODEL_DIR / "best_mlp_binary_model.keras"

# EarlyStopping stops training if validation loss stops improving
# restore_best_weights=True ensures the best weights are kept
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ModelCheckpoint saves the best model seen during training
checkpoint = ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)



# 10. Train model

# Train the network using the training data
# Validate on the validation set each epoch
# Use class weights to help with imbalance
print("Starting binary training...")
history_mlp = mlp_model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
print("Binary training complete.")
print()


# 11. Validation evaluation

# Evaluate the trained model on the validation set
print("Evaluating on validation set...")
val_results = mlp_model.evaluate(X_val_scaled, y_val, verbose=0)

# Pair metric names with metric values
metric_names = mlp_model.metrics_names
metrics_dict = dict(zip(metric_names, val_results))

print("Validation metrics:")
for k, v in metrics_dict.items():
    print(f"{k}: {v:.4f}")
print()

# Get predicted probabilities and convert them to binary predictions
y_prob_val = mlp_model.predict(X_val_scaled, verbose=0).ravel()
y_pred_val = (y_prob_val >= 0.5).astype(int)

# Compute common classification metrics
acc = accuracy_score(y_val, y_pred_val)
prec = precision_score(y_val, y_pred_val, zero_division=0)
rec = recall_score(y_val, y_pred_val, zero_division=0)
f1 = f1_score(y_val, y_pred_val, zero_division=0)
auc = roc_auc_score(y_val, y_prob_val)

print("Detailed validation metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print()

# Print the full per-class performance report
print("Validation classification report:")
print(classification_report(y_val, y_pred_val, zero_division=0))
print()

# Print confusion matrix to show correct / incorrect predictions
cm = confusion_matrix(y_val, y_pred_val)
print("Validation confusion matrix:")
print(cm)
print()

# Optional test-set evaluation
# Uncomment later if you add a separate test split
# print("Evaluating on test set...")
# test_results = mlp_model.evaluate(X_test_scaled, y_test, verbose=0)
# y_prob = mlp_model.predict(X_test_scaled, verbose=0).ravel()
# y_pred = (y_prob >= 0.5).astype(int)



# 12. Save metrics and artifacts

# Save all key results in a JSON file for later analysis
metrics_output = {
    "keras_metrics": {k: float(v) for k, v in metrics_dict.items()},
    "detailed_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    },
    "class_weight": {str(k): float(v) for k, v in class_weight.items()},
    "feature_columns": X.columns.tolist(),
    "dropped_columns": drop_cols,
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape)
    # "test_shape": list(X_test.shape)
}

metrics_file = LOG_DIR / "mlp_binary_validation_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics_output, f, indent=4)

print(f"Saved metrics to: {metrics_file}")

# Save the final trained model
final_model_path = MODEL_DIR / "final_mlp_binary_model.keras"
mlp_model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")


# 13. Save training history and plots

# Save the full training history as CSV
history_df = pd.DataFrame(history_mlp.history)
history_csv = LOG_DIR / "training_history_binary.csv"
history_df.to_csv(history_csv, index=False)
print(f"Saved training history to: {history_csv}")

# Save loss curve plot
plt.figure()
plt.plot(history_mlp.history["loss"], label="Train Loss")
plt.plot(history_mlp.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Binary Training vs Validation Loss")
plt.legend()
loss_plot_path = LOG_DIR / "loss_curve_binary.png"
plt.savefig(loss_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved loss curve to: {loss_plot_path}")

# Save accuracy curve plot if available
if "accuracy" in history_mlp.history and "val_accuracy" in history_mlp.history:
    plt.figure()
    plt.plot(history_mlp.history["accuracy"], label="Train Accuracy")
    plt.plot(history_mlp.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Binary Training vs Validation Accuracy")
    plt.legend()
    acc_plot_path = LOG_DIR / "accuracy_curve_binary.png"
    plt.savefig(acc_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve to: {acc_plot_path}")

print()
print("=" * 80)
print("Binary training pipeline finished successfully.")
print("=" * 80)