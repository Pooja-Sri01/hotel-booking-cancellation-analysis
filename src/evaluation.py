"""
Model Evaluation and Explainability Module for Hotel Booking Cancellation Analysis.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Institution: KPR Institute of Engineering and Technology
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def plot_loss_curves(history: dict, output_dir: str = OUTPUTS_DIR) -> str:
    """Plot training vs validation loss across epochs."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["loss"]) + 1)
    plt.plot(epochs, history["loss"], "b-o", linewidth=2, label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r--s", linewidth=2, label="Validation Loss")
    plt.title("Model Convergence: Training vs Validation Binary Cross-Entropy Loss", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, labelpad=8)
    plt.ylabel("Loss (Binary Cross-Entropy)", fontsize=11, labelpad=8)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eval_01_training_validation_loss.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_accuracy_curves(history: dict, output_dir: str = OUTPUTS_DIR) -> str:
    """Plot training vs validation accuracy across epochs."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["accuracy"]) + 1)
    plt.plot(epochs, history["accuracy"], "b-o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r--s", linewidth=2, label="Validation Accuracy")
    plt.title("Model Performance: Training vs Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, labelpad=8)
    plt.ylabel("Accuracy", fontsize=11, labelpad=8)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eval_02_training_validation_accuracy.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str = OUTPUTS_DIR) -> tuple[str, dict]:
    """Plot confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = [
        [f"{tn:,}\n({cm_percent[0, 0]:.1f}% TN)", f"{fp:,}\n({cm_percent[0, 1]:.1f}% FP)"],
        [f"{fn:,}\n({cm_percent[1, 0]:.1f}% FN)", f"{tp:,}\n({cm_percent[1, 1]:.1f}% TP)"],
    ]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=np.array(annot),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Fulfilled (0)", "Predicted Canceled (1)"],
        yticklabels=["Actual Fulfilled (0)", "Actual Canceled (1)"],
        annot_kws={"size": 11, "fontweight": "bold"},
    )
    plt.title("Confusion Matrix on Held-Out Test Set", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Ground Truth Outcome", fontsize=11, labelpad=8)
    plt.xlabel("Predicted Outcome", fontsize=11, labelpad=8)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eval_03_confusion_matrix.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

    cm_dict = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "tn_pct": round(float(cm_percent[0, 0]), 2),
        "fp_pct": round(float(cm_percent[0, 1]), 2),
        "fn_pct": round(float(cm_percent[1, 0]), 2),
        "tp_pct": round(float(cm_percent[1, 1]), 2),
    }
    return filepath, cm_dict


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str = OUTPUTS_DIR) -> tuple[str, float]:
    """Plot Receiver Operating Characteristic (ROC) Curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Deep Learning MLP (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eval_04_roc_curve.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath, auc_val


def calculate_permutation_importance(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    sample_size: int = 2500,
    random_state: int = 42,
    output_dir: str = OUTPUTS_DIR,
) -> tuple[pd.DataFrame, str]:
    """
    Computes permutation-based feature importance on test set sample.
    Measures degradation in ROC-AUC when each feature is independently scrambled.
    """
    print(f"[*] Calculating Permutation Feature Importance (sample size = {sample_size})...")
    np.random.seed(random_state)
    n_samples = min(sample_size, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices].copy()
    y_sample = y_test[indices]

    # Baseline ROC-AUC
    base_preds = model.predict(X_sample, verbose=0).ravel()
    base_auc = roc_auc_score(y_sample, base_preds)

    importances = []
    for col_idx in range(X_sample.shape[1]):
        X_perm = X_sample.copy()
        np.random.shuffle(X_perm[:, col_idx])
        perm_preds = model.predict(X_perm, verbose=0).ravel()
        perm_auc = roc_auc_score(y_sample, perm_preds)
        auc_drop = max(0.0, base_auc - perm_auc)
        importances.append(auc_drop)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_auc_drop": importances
    }).sort_values("importance_auc_drop", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "feature_importance.csv")
    df_imp.to_csv(csv_path, index=False)

    # Plot top 15 features
    top15 = df_imp.head(15).iloc[::-1]  # reverse for horizontal bar chart
    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(top15["feature"], top15["importance_auc_drop"], color="#2c7fb8", edgecolor="black", alpha=0.85)
    plt.title("Top 15 Most Influential Features (Permutation Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Permutation Importance Score (Decline in Test ROC-AUC)", fontsize=11, labelpad=8)
    plt.ylabel("Feature Name", fontsize=11, labelpad=8)
    plt.xlim(0, max(top15["importance_auc_drop"]) * 1.25 if max(top15["importance_auc_drop"]) > 0 else 0.1)

    for bar, val in zip(bars, top15["importance_auc_drop"]):
        plt.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "eval_05_feature_importance.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"[+] Feature importance saved to '{csv_path}' and '{chart_path}'.")
    return df_imp, chart_path


def evaluate_model_pipeline(
    model: tf.keras.Model,
    history: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: str = OUTPUTS_DIR,
) -> dict:
    """
    Executes full evaluation:
    1. Plots loss and accuracy curves.
    2. Predicts test set probabilities and class decisions.
    3. Plots confusion matrix and ROC curve.
    4. Computes exact accuracy, precision, recall, F1, and AUC.
    5. Saves metrics to CSV and JSON.
    6. Computes permutation feature importance.
    """
    print("[*] Evaluating trained Deep Learning model on held-out test set...")

    # Predictions
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.50).astype(int)

    # Evaluation Charts
    loss_fig = plot_loss_curves(history, output_dir)
    acc_fig = plot_accuracy_curves(history, output_dir)
    cm_fig, cm_dict = plot_confusion_matrix(y_test, y_pred, output_dir)
    roc_fig, auc_score = plot_roc_curve(y_test, y_prob, output_dir)

    # Performance Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    specificity = float(cm_dict["true_negatives"] / (cm_dict["true_negatives"] + cm_dict["false_positives"]))

    metrics_df = pd.DataFrame([{
        "Metric": "Accuracy", "Value": round(acc, 4), "Percentage": f"{acc*100:.2f}%"
    }, {
        "Metric": "Precision", "Value": round(prec, 4), "Percentage": f"{prec*100:.2f}%"
    }, {
        "Metric": "Recall (Sensitivity)", "Value": round(rec, 4), "Percentage": f"{rec*100:.2f}%"
    }, {
        "Metric": "Specificity", "Value": round(specificity, 4), "Percentage": f"{specificity*100:.2f}%"
    }, {
        "Metric": "F1-Score", "Value": round(f1, 4), "Percentage": f"{f1*100:.2f}%"
    }, {
        "Metric": "ROC-AUC Score", "Value": round(auc_score, 4), "Percentage": f"{auc_score*100:.2f}%"
    }])

    metrics_csv_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    cls_report = classification_report(y_test, y_pred, target_names=["Not Canceled (0)", "Canceled (1)"], output_dict=True)

    print("\n" + "=" * 60)
    print("           MODEL TEST EVALUATION METRICS")
    print("=" * 60)
    for _, row in metrics_df.iterrows():
        print(f"  {row['Metric']:<22}: {row['Value']:.4f} ({row['Percentage']})")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Canceled (0)", "Canceled (1)"]))

    # Feature Importance
    df_imp, imp_chart = calculate_permutation_importance(
        model, X_test, y_test, feature_names, output_dir=output_dir
    )

    eval_results = {
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc_score, 4),
        },
        "confusion_matrix": cm_dict,
        "classification_report": cls_report,
        "figure_paths": {
            "loss_curve": loss_fig,
            "accuracy_curve": acc_fig,
            "confusion_matrix": cm_fig,
            "roc_curve": roc_fig,
            "feature_importance": imp_chart,
        },
        "top_features": df_imp.head(10).to_dict(orient="records"),
    }

    eval_json_path = os.path.join(output_dir, "model_evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

    print(f"[+] Evaluation artifacts saved to '{output_dir}'.")
    return eval_results


if __name__ == "__main__":
    pass
