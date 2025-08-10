"""
Train a baseline expense categorizer and track everything with MLFlow.

What this does:
- Loads + validates the data (pandera).
- Time-based split (last 14 days = test).
- Builds a TF-IDF + LogisticRegression pipeline.
- Trains, evaluates, saves a classification report + confusion matrix image.
- Logs params/metrics/artifacts + the model to MLFlow.

Run:
    python -m scripts.train_baseline_mlflow data/transactions.csv artifacts/model.joblib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.validation.schema import TransactionsSchema


def load_and_validate(path: str) -> pd.DataFrame:
    """Loads CSV, drop amount == 0, validate with Pandera"""
    df = pd.read_csv(path)
    df = df[df["amount"].astype(float) > 0].copy()
    TransactionsSchema.validate(df, lazy=True)
    return df


def time_based_split(df: pd.DataFrame, test_days: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so test is the most recent `test_days`."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()
    return train_df, test_df


def combine_text_df(X: pd.DataFrame) -> pd.Series:
    """
    Top-level (picklable) function that combines text columns.

    Returns a 1D sequence of strings like:
        "uber ride to office uber"
    """
    return (X["description"].fillna("") + " " + X["merchant"].fillna("")).astype(str)


def build_pipeline(ngram_max: int = 2, max_features: int = 20000) -> Pipeline:
    preproc = ColumnTransformer(
        transformers=[
            (
                "text",
                Pipeline(
                    [
                        ("combine", FunctionTransformer(combine_text_df, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features),
                        ),
                    ]
                ),
                ["description", "merchant"],
            ),
            ("amount", StandardScaler(), ["amount"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(max_iter=1000)
    return Pipeline(steps=[("preproc", preproc), ("clf", clf)])


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, out_path: str) -> None:
    """Plot and save a confusion matrix image."""

    labels = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main(data_path: str, model_path: str) -> None:
    # Configure MLflow (local folder by default)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("expense-autocategorizer-baseline")

    df = load_and_validate(data_path)
    train_df, test_df = time_based_split(df, test_days=14)

    X_train, y_train = train_df[["description", "merchant", "amount"]], train_df["label"]
    X_test, y_test = test_df[["description", "merchant", "amount"]], test_df["label"]

    # Hyperparams you want tracked
    ngram_max = 2
    max_features = 20000

    with mlflow.start_run():
        mlflow.log_params(
            {
                "vectorizer": "tfidf",
                "ngram_max": ngram_max,
                "max_features": max_features,
                "clf": "logistic_regression",
                "split_test_days": 14,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        pipe = build_pipeline(ngram_max=ngram_max, max_features=max_features)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("macro_f1", float(macro_f1))

        # Per-class F1 (nice for debugging)
        labels = sorted(pd.unique(y_test))
        per_class = f1_score(y_test, y_pred, average=None, labels=labels)
        for lbl, f1 in zip(labels, per_class):
            mlflow.log_metric(f"f1_{lbl}", float(f1))

        # Save artifacts: report + confusion matrix
        report_txt = classification_report(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        mlflow.log_artifact(report_path)

        cm_path = "artifacts/confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        # Log the macro F1 score
        mlflow.log_metric("macro_f1", float(macro_f1))

        # NEW: record the gate threshold + pass/fail in MLflow
        THRESHOLD = 0.80
        mlflow.log_param("quality_gate_threshold", THRESHOLD)

        gate_status = "pass" if macro_f1 >= THRESHOLD else "fail"
        mlflow.set_tag("quality_gate", gate_status)
        mlflow.set_tag("stage", "training")  # optional, helps filtering in UI

        # Enforce the gate
        if macro_f1 < THRESHOLD:
            print(f"❌ Macro-F1 below threshold ({THRESHOLD}): {macro_f1:.3f}")
            import sys

            sys.exit(1)
        else:
            print(f"✅ Macro-F1 passes threshold ({THRESHOLD}): {macro_f1:.3f}")

        # Persist model both to disk and MLflow
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=None)

        print(report_txt)
        print(f"\nMacro-F1: {macro_f1:.3f}")
        print(f"Model saved to {model_path}")
        print("Run logged to MLflow.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m scripts.train_baseline_mlflow data/transactions.csv artifacts/model.joblib"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
