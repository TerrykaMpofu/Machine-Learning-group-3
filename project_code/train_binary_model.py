# train_binary_model.py
# Binary CAN intrusion detection training script
# This script:
# 1. Loads the cleaned binary CAN dataset
# 2. Removes unnecessary / leakage columns
# 3. Cleans NaN and infinite values
# 4. Splits the data into train / validation / test sets
# 5. Scales the numeric features
# 6. Builds and trains a tuned MLP neural network
# 7. Tunes the classification threshold on validation data
# 8. Evaluates performance on the test set
# 9. Saves the trained model, scaler, test data, metrics, and plots


from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN,
    ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2

# 1. Basic settings

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (10, 5)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path.home() / "projects" / "can_ids"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "binary"
LOG_DIR = BASE_DIR / "logs" / "binary"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BINARY_FILE = DATA_DIR / "can_binary_cleaned.csv"

print("=" * 80)
print("CAN IDS Binary Neural Network Training")
print("=" * 80)
print(f"Base directory      : {BASE_DIR}")
print(f"Data directory      : {DATA_DIR}")
print(f"Binary dataset path : {BINARY_FILE}")
print("GPUs available      :", tf.config.list_physical_devices("GPU"))
print()

# 2. Check file exists
if not BINARY_FILE.exists():
    raise FileNotFoundError(
        f"Binary dataset not found: {BINARY_FILE}\n"
        "Make sure your dataset is in ~/projects/can_ids/data/"
    )

# 3. Load dataset
binary_df = pd.read_csv(BINARY_FILE)

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
df = binary_df.copy()

if "binary_target" not in df.columns:
    raise ValueError("Expected column 'binary_target' not found in binary dataset.")

y = df["binary_target"].copy()

drop_cols = [
    "binary_target",
    "binary_label",
    "class_label",
    "source_file",
    "direction",
    "timestamp"
]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=[np.number]).copy()

if X.empty:
    raise ValueError("No numeric feature columns found after preprocessing.")

print("Initial feature columns used for binary training:")
print(X.columns.tolist())
print()
print(f"Initial X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Target distribution:")
print(y.value_counts(dropna=False))
print()

# Clean target column
valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()

y = y.astype(int)

binary_mask = y.isin([0, 1])
X = X.loc[binary_mask].copy()
y = y.loc[binary_mask].copy()

print("Target distribution after target cleanup:")
print(y.value_counts(dropna=False))
print()

# Inspect NaN and inf before cleaning
print("Missing values per feature before cleaning:")
print(X.isna().sum().sort_values(ascending=False))
print()

numeric_array = X.to_numpy(dtype=float)
print("Total inf values before cleaning:", np.isinf(numeric_array).sum())
print("Total NaN values before cleaning:", np.isnan(numeric_array).sum())
print()

# Clean feature matrix
X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)

feature_medians = X.median(numeric_only=True)
X = X.fillna(feature_medians)
X = X.fillna(0)

# Safer clipping for extreme outliers
X = X.clip(lower=-1e6, upper=1e6)

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

print("Feature summary statistics:")
feature_summary = X.describe().T[["mean", "std", "min", "max"]].sort_values(by="std", ascending=False)
print(feature_summary.head(20))
print()

print("Top features with largest absolute values:")
max_abs = X.abs().max().sort_values(ascending=False)
print(max_abs.head(20))
print()


# 5. Train / validation / test split (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

print("Data split complete:")
print(f"Train      : {X_train.shape}, {y_train.shape}")
print(f"Validation : {X_val.shape}, {y_val.shape}")
print(f"Test       : {X_test.shape}, {y_test.shape}")
print()


# 6. Scale features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaling complete:")
print(f"Scaled train shape: {X_train_scaled.shape}")
print(f"Scaled val shape  : {X_val_scaled.shape}")
print(f"Scaled test shape : {X_test_scaled.shape}")
print()

print("Post-scaling sanity checks:")
print("NaN in X_train_scaled:", np.isnan(X_train_scaled).any())
print("NaN in X_val_scaled  :", np.isnan(X_val_scaled).any())
print("NaN in X_test_scaled :", np.isnan(X_test_scaled).any())
print("Inf in X_train_scaled:", np.isinf(X_train_scaled).any())
print("Inf in X_val_scaled  :", np.isinf(X_val_scaled).any())
print("Inf in X_test_scaled :", np.isinf(X_test_scaled).any())
print("y_train unique values:", np.unique(y_train))
print("y_val unique values  :", np.unique(y_val))
print("y_test unique values :", np.unique(y_test))
print()

if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any() or np.isnan(X_test_scaled).any():
    raise ValueError("NaN values detected after scaling. Stop and inspect preprocessing.")

if np.isinf(X_train_scaled).any() or np.isinf(X_val_scaled).any() or np.isinf(X_test_scaled).any():
    raise ValueError("Inf values detected after scaling. Stop and inspect preprocessing.")

# Save scaler
scaler_path = MODEL_DIR / "binary_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Saved scaler to: {scaler_path}")

# Save held-out test set
test_data_path = MODEL_DIR / "binary_test_data.npz"
np.savez_compressed(
    test_data_path,
    X_test_scaled=X_test_scaled,
    y_test=y_test.to_numpy(),
    feature_names=np.array(X.columns.tolist(), dtype=object)
)
print(f"Saved test data to: {test_data_path}")
print()



# 7. Build tuned binary MLP model
# Output bias can help with class imbalance
positive_rate = y_train.mean()
initial_bias = np.log(positive_rate / (1.0 - positive_rate))

mlp_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu", kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    Dropout(0.30),

    Dense(32, activation="relu", kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    Dropout(0.25),

    Dense(16, activation="relu", kernel_regularizer=l2(1e-5)),
    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(initial_bias)
    )
])

mlp_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-5,
        clipnorm=1.0
    ),
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

# 8. Callbacks

best_model_path = MODEL_DIR / "best_mlp_binary_model.keras"

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

nan_stop = TerminateOnNaN()


# 9. Train model

print("Starting binary training...")
history_mlp = mlp_model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=40,
    batch_size=64,
    callbacks=[early_stop, checkpoint, reduce_lr, nan_stop],
    verbose=1
)
print("Binary training complete.")
print()

# 10. Tune threshold on validation set

print("Tuning classification threshold on validation set...")

y_prob_val = mlp_model.predict(X_val_scaled, verbose=0).ravel()

thresholds = np.arange(0.10, 0.91, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_val = (y_prob_val >= threshold).astype(int)

    val_acc = accuracy_score(y_val, y_pred_val)
    val_prec = precision_score(y_val, y_pred_val, zero_division=0)
    val_rec = recall_score(y_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
    val_bal_acc = balanced_accuracy_score(y_val, y_pred_val)

    cm_val = confusion_matrix(y_val, y_pred_val)
    tn, fp, fn, tp = cm_val.ravel()
    val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    threshold_results.append({
        "threshold": float(threshold),
        "accuracy": float(val_acc),
        "precision": float(val_prec),
        "recall": float(val_rec),
        "f1_score": float(val_f1),
        "balanced_accuracy": float(val_bal_acc),
        "specificity": float(val_specificity)
    })

threshold_df = pd.DataFrame(threshold_results)

# Choose threshold using balanced accuracy first, then F1
best_row = threshold_df.sort_values(
    by=["balanced_accuracy", "f1_score"],
    ascending=[False, False]
).iloc[0]

best_threshold = float(best_row["threshold"])

print("Validation threshold tuning results:")
print(threshold_df)
print()
print(f"Best threshold selected: {best_threshold:.2f}")
print("Best validation metrics at selected threshold:")
print(best_row.to_dict())
print()

threshold_csv_path = LOG_DIR / "threshold_tuning_binary.csv"
threshold_df.to_csv(threshold_csv_path, index=False)
print(f"Saved threshold tuning table to: {threshold_csv_path}")

threshold_json_path = LOG_DIR / "best_threshold_binary.json"
with open(threshold_json_path, "w") as f:
    json.dump(
        {
            "best_threshold": best_threshold,
            "selection_metric": "balanced_accuracy_then_f1",
            "best_validation_metrics": {k: float(v) for k, v in best_row.to_dict().items()}
        },
        f,
        indent=4
    )
print(f"Saved best threshold to: {threshold_json_path}")
print()


# 11. Test evaluation using tuned threshold

print("Evaluating on test set...")
test_results = mlp_model.evaluate(X_test_scaled, y_test, verbose=0)

metric_names = mlp_model.metrics_names
metrics_dict = dict(zip(metric_names, test_results))

print("Test metrics from model.evaluate():")
for k, v in metrics_dict.items():
    print(f"{k}: {v:.4f}")
print()

y_prob_test = mlp_model.predict(X_test_scaled, verbose=0).ravel()
y_pred_test = (y_prob_test >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)
auc = roc_auc_score(y_test, y_prob_test)
bal_acc = balanced_accuracy_score(y_test, y_pred_test)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print("Detailed test metrics:")
print(f"Threshold         : {best_threshold:.2f}")
print(f"Accuracy          : {acc:.4f}")
print(f"Precision         : {prec:.4f}")
print(f"Recall            : {rec:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"ROC AUC           : {auc:.4f}")
print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"Specificity       : {specificity:.4f}")
print()

print("Test classification report:")
print(classification_report(y_test, y_pred_test, zero_division=0))
print()

print("Test confusion matrix:")
print(cm)
print()


# 12. Save metrics and artifacts

metrics_output = {
    "keras_metrics": {k: float(v) for k, v in metrics_dict.items()},
    "threshold_used": float(best_threshold),
    "detailed_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "balanced_accuracy": float(bal_acc),
        "specificity": float(specificity)
    },
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    },
    "feature_columns": X.columns.tolist(),
    "dropped_columns": drop_cols,
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape),
    "test_shape": list(X_test.shape)
}

metrics_file = LOG_DIR / "mlp_binary_test_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics_output, f, indent=4)

print(f"Saved metrics to: {metrics_file}")

final_model_path = MODEL_DIR / "final_mlp_binary_model.keras"
mlp_model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")


# 13. Save training history and plots

history_df = pd.DataFrame(history_mlp.history)
history_csv = LOG_DIR / "training_history_binary.csv"
history_df.to_csv(history_csv, index=False)
print(f"Saved training history to: {history_csv}")

# Loss curve
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

# Accuracy curve
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

# Precision curve
if "precision" in history_mlp.history and "val_precision" in history_mlp.history:
    plt.figure()
    plt.plot(history_mlp.history["precision"], label="Train Precision")
    plt.plot(history_mlp.history["val_precision"], label="Validation Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Binary Training vs Validation Precision")
    plt.legend()
    precision_plot_path = LOG_DIR / "precision_curve_binary.png"
    plt.savefig(precision_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved precision curve to: {precision_plot_path}")

# Recall curve
if "recall" in history_mlp.history and "val_recall" in history_mlp.history:
    plt.figure()
    plt.plot(history_mlp.history["recall"], label="Train Recall")
    plt.plot(history_mlp.history["val_recall"], label="Validation Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Binary Training vs Validation Recall")
    plt.legend()
    recall_plot_path = LOG_DIR / "recall_curve_binary.png"
    plt.savefig(recall_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved recall curve to: {recall_plot_path}")

# AUC curve
if "auc" in history_mlp.history and "val_auc" in history_mlp.history:
    plt.figure()
    plt.plot(history_mlp.history["auc"], label="Train AUC")
    plt.plot(history_mlp.history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Binary Training vs Validation AUC")
    plt.legend()
    auc_plot_path = LOG_DIR / "auc_curve_binary.png"
    plt.savefig(auc_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved AUC curve to: {auc_plot_path}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Binary ROC Curve (Test Set)")
plt.legend()
roc_plot_path = LOG_DIR / "roc_curve_binary.png"
plt.savefig(roc_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved ROC curve to: {roc_plot_path}")

# Precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
plt.figure()
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Binary Precision-Recall Curve (Test Set)")
pr_plot_path = LOG_DIR / "precision_recall_curve_binary.png"
plt.savefig(pr_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved precision-recall curve to: {pr_plot_path}")

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title(f"Binary Confusion Matrix (Test Set, threshold={best_threshold:.2f})")
cm_plot_path = LOG_DIR / "confusion_matrix_binary.png"
plt.savefig(cm_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix plot to: {cm_plot_path}")

print()
print("=" * 80)
print("Binary training pipeline finished successfully.")
print("=" * 80)
