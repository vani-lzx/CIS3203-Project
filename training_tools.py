from .chart_tools import create_model_charts
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis_tools import DATA_DIR, inspect_dataset


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Calculate test metrics with class 1 as the positive class."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }


def train_classifier(
    filename: str,
    target_column: str,
    feature_columns: list[str],
    failure_threshold: float = 10.0,
    separator: str = ";",
) -> dict:
    """Train and evaluate a classifier using a numeric target threshold.

    Args:
        filename: CSV filename inside the data directory.
        target_column: Numeric outcome column.
        feature_columns: Input columns, excluding the target.
        failure_threshold: Values below this threshold become class 1.
        separator: CSV delimiter.

    Returns:
        Dataset summary, test metrics, and saved model location.
        Class 1 means below threshold. Class 0 means at or above threshold.
    """
    summary = inspect_dataset(
        filename, target_column, feature_columns, separator
    )
    if "error" in summary:
        return summary

    if not np.isfinite(failure_threshold):
        return {"error": "The threshold must be a finite number."}

    df = pd.read_csv(DATA_DIR / filename, sep=separator)
    target = pd.to_numeric(df[target_column], errors="coerce")

    if target.isna().any() or not np.isfinite(target).all():
        return {"error": "The target must contain only finite numeric values."}

    X = df[feature_columns].copy()
    y = (target < failure_threshold).astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 5:
        return {"error": "Each target class must contain at least 5 rows."}

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", keep_empty_features=True
        )),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent", keep_empty_features=True
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model.fit(X_train, y_train)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)

        model_metrics = evaluate_classifier(model, X_test, y_test)
        baseline_metrics = evaluate_classifier(baseline, X_test, y_test)
    except ValueError as error:
        return {"error": str(error)}

    output_dir = (
        Path(__file__).resolve().parent / "outputs" / uuid4().hex
    )
    output_dir.mkdir(parents=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)

    result = {
        "status": "success",
        "dataset_summary": summary,
        "experiment": "Pooled stratified train-test split",
        "model": "Logistic Regression",
        "class_weight": "balanced",
        "positive_class": f"{target_column} < {failure_threshold}",
        "negative_class": f"{target_column} >= {failure_threshold}",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_positive_rows": int(y_test.sum()),
        "random_state": 42,
        "confusion_matrix_layout": [
            ["true_negative", "false_positive"],
            ["false_negative", "true_positive"],
        ],
        "model_metrics": model_metrics,
        "majority_baseline_metrics": baseline_metrics,
        "model_path": str(model_path),
    }

    result.update(
        create_model_charts(model, X_test, y_test, output_dir)
    )

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
