"""Evaluation pipeline — loads the latest registered model and prints a full report."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "fraud-detector"
MODEL_STAGE = os.getenv("MODEL_STAGE", "None")  # "None" | "Staging" | "Production"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_splits() -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_test, y_test


def load_model(tracking_uri: str):
    """Load the latest version of the registered model from MLflow Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def evaluate() -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model = load_model(tracking_uri)

    X_test, y_test = load_test_splits()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"Model : {MODEL_NAME}  (stage={MODEL_STAGE})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("=" * 60)

    return {"f1": f1, "roc_auc": roc_auc, "precision": precision, "recall": recall}


if __name__ == "__main__":
    evaluate()
