"""Training pipeline — GradientBoostingClassifier baseline with full MLflow tracking."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in CI / Docker
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")
MODEL_NAME = "fraud-detector"
MIN_F1_THRESHOLD = float(os.getenv("MIN_F1_THRESHOLD", "0.85"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def build_model(params: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        random_state=params["random_state"],
    )


def _save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, tmp_dir: str) -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    path = os.path.join(tmp_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _save_feature_importance(
    model: GradientBoostingClassifier, feature_names: list[str], tmp_dir: str
) -> str:
    importances = model.feature_importances_
    top_n = min(20, len(feature_names))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[indices], importances[indices])
    ax.set_title(f"Feature Importances (top {top_n})")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    path = os.path.join(tmp_dir, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def quality_gate(f1: float) -> None:
    """Raise ValueError if F1 is below the minimum threshold."""
    if f1 < MIN_F1_THRESHOLD:
        raise ValueError(
            f"Quality gate failed: F1={f1:.4f} < threshold={MIN_F1_THRESHOLD}. "
            "Model will NOT be registered."
        )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn-prediction")

    params = load_params()
    X_train, X_test, y_train, y_test = load_splits()

    with mlflow.start_run():
        # --- fit ---
        model = build_model(params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --- metrics ---
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics(
            {"f1": f1, "roc_auc": roc_auc, "precision": precision, "recall": recall}
        )

        # --- artifacts ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.log_artifact(
                _save_confusion_matrix(y_test, y_pred, tmp_dir)
            )
            mlflow.log_artifact(
                _save_feature_importance(model, X_train.columns.tolist(), tmp_dir)
            )

        # --- report ---
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        print(f"ROC-AUC : {roc_auc:.4f}")
        print(f"F1      : {f1:.4f}")

        # --- quality gate (must pass before registration) ---
        quality_gate(f1)

        # --- register ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        print(f"Model registered as '{MODEL_NAME}' in MLflow Registry.")


if __name__ == "__main__":
    train()
