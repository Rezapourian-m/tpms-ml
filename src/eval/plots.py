from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_training(history, title_prefix="Model"):
    # Training/validation loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title(f"{title_prefix} - Loss"); plt.legend(); plt.tight_layout()
    plt.show()

def parity_plot(y_true, y_pred, labels=None, title="Parity"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = y_true.shape[1] if y_true.ndim == 2 else 1
    for i in range(n):
        plt.figure(figsize=(4,4))
        a = y_true[:, i] if n>1 else y_true
        b = y_pred[:, i] if n>1 else y_pred
        plt.scatter(a, b, s=12, alpha=0.7)
        lo, hi = float(np.min(a)), float(np.max(a))
        plt.plot([lo, hi],[lo, hi], "r--")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(labels[i] if labels else title)
        plt.tight_layout(); plt.show()
