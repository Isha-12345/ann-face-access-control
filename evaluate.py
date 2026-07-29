"""
evaluate.py
-----------
Evaluates a trained classifier on the test split and produces the figures
and tables required by the report.

Usage:
    python evaluate.py --model mobilenetv2
    python evaluate.py --compare          # build comparison table of all models

Outputs:
    results/<name>_confusion.png
    results/<name>_roc.png
    results/<name>_metrics.json
    results/model_comparison.csv
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)

import config as C


def load_test_ds():
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(C.DATA_DIR, "test"),
        labels="inferred", label_mode="categorical",
        class_names=C.CLASS_NAMES,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE, shuffle=False,
    )


def predict(model, ds):
    y_true, y_prob = [], []
    for xb, yb in ds:
        y_prob.append(model.predict(xb, verbose=0))
        y_true.append(yb.numpy())
    return np.concatenate(y_true).argmax(1), np.concatenate(y_prob)


def security_metrics(y_true, y_pred):
    """FAR: a non-authorised person accepted as known (the dangerous error).
       FRR: an authorised person rejected."""
    k = C.KNOWN_INDEX
    non_known = y_true != k
    known = y_true == k
    far = float((y_pred[non_known] == k).mean()) if non_known.any() else 0.0
    frr = float((y_pred[known] != k).mean()) if known.any() else 0.0
    return far, frr


def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASS_NAMES)), C.CLASS_NAMES, rotation=20)
    ax.set_yticks(range(len(C.CLASS_NAMES)), C.CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix - {name}")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(C.RESULT_DIR, f"{name}_confusion.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[i] saved {out}")


def plot_roc(y_true, y_prob, name):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    aucs = {}
    for i, cls in enumerate(C.CLASS_NAMES):
        binary = (y_true == i).astype(int)
        if binary.min() == binary.max():
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        auc = roc_auc_score(binary, y_prob[:, i])
        aucs[cls] = float(auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curves (one-vs-rest) - {name}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(C.RESULT_DIR, f"{name}_roc.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[i] saved {out}")
    return aucs


def evaluate_one(name):
    path = os.path.join(C.MODEL_DIR, f"{name}.keras")
    if not os.path.exists(path):
        raise SystemExit(f"[!] Model not found: {path}")
    model = tf.keras.models.load_model(path)
    ds = load_test_ds()
    y_true, y_prob = predict(model, ds)
    y_pred = y_prob.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    far, frr = security_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(C.CLASS_NAMES)))

    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, target_names=C.CLASS_NAMES,
                                zero_division=0))
    print(f"accuracy {acc:.4f} | macro-F1 {f1:.4f} | FAR {far:.4f} | FRR {frr:.4f}")

    plot_confusion(cm, name)
    aucs = plot_roc(y_true, y_prob, name)

    metrics = {
        "model": name, "accuracy": float(acc), "precision_macro": float(prec),
        "recall_macro": float(rec), "f1_macro": float(f1),
        "FAR": far, "FRR": frr, "auc_per_class": aucs,
        "confusion_matrix": cm.tolist(),
        "class_names": C.CLASS_NAMES,
        "n_test_images": int(len(y_true)),
    }
    with open(os.path.join(C.RESULT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def build_comparison():
    rows = []
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, "*_metrics.json"))):
        if os.path.basename(p) == "embedding_metrics.json":
            continue
        with open(p) as f:
            m = json.load(f)
        rows.append({
            "Model": m["model"], "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision_macro"], 4),
            "Recall": round(m["recall_macro"], 4),
            "F1": round(m["f1_macro"], 4),
            "FAR": round(m["FAR"], 4), "FRR": round(m["FRR"], 4),
        })

    emb = os.path.join(C.RESULT_DIR, "embedding_metrics.json")
    if os.path.exists(emb):
        with open(emb) as f:
            e = json.load(f)
        rows.append({
            "Model": "embedding (open-set)",
            "Accuracy": round(e["test_accuracy_open_set"], 4),
            "Precision": "-", "Recall": "-", "F1": "-",
            "FAR": round(e["test_FAR"], 4), "FRR": round(e["test_FRR"], 4),
        })

    if not rows:
        print("[!] No metrics found. Run evaluate.py --model <name> first.")
        return
    df = pd.DataFrame(rows)
    out = os.path.join(C.RESULT_DIR, "model_comparison.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[i] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mobilenetv2", "mobilenetv3", "baseline"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.model:
        evaluate_one(args.model)
    if args.compare or not args.model:
        build_comparison()


if __name__ == "__main__":
    main()
