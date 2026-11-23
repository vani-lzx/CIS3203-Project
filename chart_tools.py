from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay


def create_model_charts(model, X_test, y_test, output_dir) -> dict:
    """Create test-set charts and permutation feature importance."""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        labels=[0, 1],
        display_labels=["At/above threshold", "Below threshold"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Test Set Confusion Matrix")
    fig.tight_layout()

    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
        n_jobs=1,
    )

    feature_results = [
        {
            "feature": str(column),
            "mean_auc_decrease": float(mean),
            "std_auc_decrease": float(std),
        }
        for column, mean, std in zip(
            X_test.columns,
            importance.importances_mean,
            importance.importances_std,
        )
    ]
    feature_results.sort(
        key=lambda item: item["mean_auc_decrease"],
        reverse=True,
    )

    plot_results = feature_results[:15][::-1]
    fig, ax = plt.subplots(
        figsize=(8, max(4, len(plot_results) * 0.4))
    )
    ax.barh(
        [item["feature"] for item in plot_results],
        [item["mean_auc_decrease"] for item in plot_results],
        xerr=[item["std_auc_decrease"] for item in plot_results],
        color="#377EB8",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean decrease in test ROC AUC")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()

    importance_path = output_dir / "feature_importance.png"
    fig.savefig(importance_path, dpi=160)
    plt.close(fig)

    return {
        "feature_importance_method": "Test-set permutation importance",
        "feature_importance_scoring": "ROC AUC",
        "feature_importance_repeats": 30,
        "feature_importance": feature_results,
        "charts": {
            "confusion_matrix": str(confusion_path),
            "feature_importance": str(importance_path),
        },
    }
