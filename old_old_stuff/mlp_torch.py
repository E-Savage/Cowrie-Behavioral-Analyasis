# ========================================
# mlp_torch.py — full metrics + batching + standardized + grid confusion matrix
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ===== LOAD DATA =====
df = pd.read_csv("datasets/ip_entropies_with_kmeans.csv")
df = df.drop(columns=["src_ip"])

df_std = df[
    ["total_sessions", "username_entropy", "password_entropy", "command_entropy"]
].fillna(0)

# ===== STANDARDIZATION =====
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_std)
standardized_data = pd.DataFrame(scaled_values, columns=df_std.columns)

standardized_data["cluster"] = df["kmeans_cluster"].astype("category").cat.codes

X = standardized_data[
    ["total_sessions", "username_entropy", "password_entropy", "command_entropy"]
]
y = standardized_data["cluster"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

input_dim = X_train.shape[1]
num_classes = int(y_train.max() + 1)


# ===== CLASS WEIGHTS =====
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ===== MODEL =====
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ===== DATALOADERS =====
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)


# ===== METRIC STORAGE =====
epochs = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []
f1_scores = []
recall_class2, recall_class3 = [], []


# ===== TRAINING LOOP =====
for epoch in range(epochs):

    # ---- TRAIN ----
    model.train()
    batch_losses = []
    correct, total = 0, 0

    for Xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    train_loss = np.mean(batch_losses)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)


    # ---- VALIDATION ----
    model.eval()
    val_losses_epoch = []
    v_correct, v_total = 0, 0
    v_preds, v_true = [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            val_losses_epoch.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            v_correct += (preds == yb).sum().item()
            v_total += yb.size(0)

            v_preds.extend(preds.cpu().numpy())
            v_true.extend(yb.cpu().numpy())

    val_loss = np.mean(val_losses_epoch)
    val_acc = v_correct / v_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # F1 + minority recall
    f1_scores.append(f1_score(v_true, v_preds, average="macro"))
    recall_class2.append(
        recall_score(v_true, v_preds, labels=[2], average="macro", zero_division=0)
    )
    recall_class3.append(
        recall_score(v_true, v_preds, labels=[3], average="macro", zero_division=0)
    )

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
        )


# ===== FINAL CONFUSION MATRIX =====
model.eval()
final_preds = []

with torch.no_grad():
    for Xb, _ in test_loader:
        final_preds.extend(torch.argmax(model(Xb), dim=1).cpu().numpy())

cm = confusion_matrix(y_test, final_preds)

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

plt.title("PyTorch Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_torch.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n===== PYTORCH RESULTS =====")
print(classification_report(y_test, final_preds))


# ===== PLOTS =====

# Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Torch Loss Curve")
plt.grid(True)
plt.savefig("loss_curve_torch.png", dpi=150)
plt.close()

# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.legend()
plt.title("Torch Accuracy Curve")
plt.grid(True)
plt.savefig("accuracy_curve_torch.png", dpi=150)
plt.close()

# F1
plt.figure(figsize=(10, 5))
plt.plot(f1_scores, label="Macro F1 Score")
plt.legend()
plt.title("Torch F1 Curve")
plt.grid(True)
plt.savefig("f1_curve_torch.png", dpi=150)
plt.close()

# Minority recall
plt.figure(figsize=(10, 5))
plt.plot(recall_class2, label="Recall Class 2")
plt.plot(recall_class3, label="Recall Class 3")
plt.legend()
plt.title("Torch Minority-Class Recall")
plt.grid(True)
plt.savefig("minority_recall_curve_torch.png", dpi=150)
plt.close()
