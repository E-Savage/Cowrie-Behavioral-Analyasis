# ========================================
# mlp_tensorflow.py — full metrics + standardized + grid confusion matrix
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models


# ===== LOAD DATA =====
df = pd.read_csv("datasets/ip_entropies_with_kmeans.csv")
df = df.drop(columns=["src_ip"])

df_std = df[
    ["total_sessions", "username_entropy", "password_entropy", "command_entropy"]
].fillna(0)

# ===== STANDARDIZATION =====
scaler = StandardScaler()
scaled = scaler.fit_transform(df_std)
standardized = pd.DataFrame(scaled, columns=df_std.columns)

standardized["cluster"] = df["kmeans_cluster"].astype("category").cat.codes

X = standardized[
    ["total_sessions", "username_entropy", "password_entropy", "command_entropy"]
]
y = standardized["cluster"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_dim = X_train.shape[1]
num_classes = int(y_train.max() + 1)


# ===== MODEL =====
model = models.Sequential(
    [
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ===== TRAIN =====
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1,
)


# ===== FINAL PREDICTIONS =====
pred_prob = model.predict(X_test)
preds = pred_prob.argmax(axis=1)


# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="black"
)

ax.set_xticks(np.arange(cm.shape[1]) + 0.5, minor=True)
ax.set_yticks(np.arange(cm.shape[0]) + 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

plt.title("TensorFlow Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_tf.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n===== TENSORFLOW RESULTS =====")
print(classification_report(y_test, preds))


# ===== METRICS =====
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

# Static curves (TF does not give per-epoch preds)
f1_macro = f1_score(y_test, preds, average="macro")
recall2 = recall_score(y_test, preds, labels=[2], average="macro", zero_division=0)
recall3 = recall_score(y_test, preds, labels=[3], average="macro", zero_division=0)


# ===== PLOTS =====

# Loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("TF Loss Curve")
plt.grid(True)
plt.savefig("loss_curve_tf.png", dpi=150)
plt.close()

# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("TF Accuracy Curve")
plt.grid(True)
plt.savefig("accuracy_curve_tf.png", dpi=150)
plt.close()

# F1 Score (static)
plt.figure(figsize=(10, 5))
plt.plot([f1_macro] * 50, label="Macro F1 Score")
plt.legend()
plt.title("TF F1 Score")
plt.grid(True)
plt.savefig("f1_curve_tf.png", dpi=150)
plt.close()

# Minority Recall (static)
plt.figure(figsize=(10, 5))
plt.plot([recall2] * 50, label="Recall Class 2")
plt.plot([recall3] * 50, label="Recall Class 3")
plt.legend()
plt.title("TF Minority-Class Recall")
plt.grid(True)
plt.savefig("minority_recall_curve_tf.png", dpi=150)
plt.close()
 