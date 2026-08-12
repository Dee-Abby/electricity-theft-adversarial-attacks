"""
Report visualization functions. Run each against your actual notebook objects
(y_test, y_proba, best_rf, results dicts from evaluate_attack, etc.) - no
hardcoded numbers, everything is computed from what you already have.

All functions save a PNG (dpi=200) suitable for pasting into the Word report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_auc_score
)


# ---------------------------------------------------------------------------
# 1. Baseline vs Tuned model comparison
# ---------------------------------------------------------------------------
def plot_baseline_vs_tuned(y_test, y_pred_baseline, y_proba_baseline,
                            y_pred_tuned, y_proba_tuned,
                            save_path="baseline_vs_tuned.png"):
    """
    Grouped bar chart comparing precision/recall/F1/AUC (theft class) between
    your untuned baseline RF and the hyperparameter-tuned RF.
    """
    def metrics(y_pred, y_proba):
        return [
            precision_score(y_test, y_pred, pos_label=1),
            recall_score(y_test, y_pred, pos_label=1),
            f1_score(y_test, y_pred, pos_label=1),
            roc_auc_score(y_test, y_proba),
        ]

    labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    baseline_vals = metrics(y_pred_baseline, y_proba_baseline)
    tuned_vals = metrics(y_pred_tuned, y_proba_tuned)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline RF", color="#4C72B0")
    b2 = ax.bar(x + width/2, tuned_vals, width, label="Tuned RF", color="#DD8452")

    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs. Tuned Random Forest \u2014 Theft Class Performance")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# 2. Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png", title="Confusion Matrix (Tuned RF)"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Legitimate", "Theft"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# 3. ROC curve + Precision-Recall curve (side by side)
# ---------------------------------------------------------------------------
def plot_roc_pr_curves(y_test, y_proba, save_path="roc_pr_curves.png"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    theft_prevalence = y_test.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax1 = axes[0]
    ax1.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    ax2 = axes[1]
    ax2.plot(recall, precision, color="#C44E52", linewidth=2, label="Precision-Recall")
    ax2.axhline(theft_prevalence, color="gray", linestyle="--", linewidth=1,
                label=f"No-skill baseline ({theft_prevalence:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (Theft Class)")
    ax2.legend(loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# 4. Feature importance chart
# ---------------------------------------------------------------------------
def plot_feature_importance(importances, save_path="feature_importance.png"):
    """
    importances: pandas Series (feature name -> importance), e.g. from
    evaluate_rf()'s returned results['importances'], or
    pd.Series(rf.feature_importances_, index=X_train.columns)
    """
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(importances.index, importances.values, color="#55A868")
    for b, v in zip(bars, importances.values):
        ax.annotate(f"{v:.3f}", (v, b.get_y() + b.get_height()/2), va="center", fontsize=9,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. Theft-probability shift before/after an attack
# ---------------------------------------------------------------------------
def plot_probability_shift(y_proba_before, y_proba_after, attack_name,
                            save_path=None, threshold=0.5):
    """
    y_proba_before: theft-probability scores for theft customers BEFORE attack
                    (e.g. best_rf.predict_proba(X_test)[:,1] filtered to theft rows)
    y_proba_after:  same customers' scores AFTER perturbation
                    (e.g. results['y_proba'] returned by evaluate_attack)
    """
    if save_path is None:
        save_path = f"probability_shift_{attack_name.lower().replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 26)
    ax.hist(y_proba_before, bins=bins, alpha=0.6, label="Before attack", color="#4C72B0")
    ax.hist(y_proba_after, bins=bins, alpha=0.6, label="After attack", color="#C44E52")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"Decision threshold ({threshold})")
    ax.set_xlabel("Theft Probability Score")
    ax.set_ylabel("Number of Theft Customers")
    ax.set_title(f"Theft-Probability Shift: {attack_name}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    print("Import these into your notebook: plot_baseline_vs_tuned, plot_confusion_matrix, "
          "plot_roc_pr_curves, plot_feature_importance, plot_probability_shift")