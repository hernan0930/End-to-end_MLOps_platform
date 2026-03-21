"""Tests for pipelines/train.py — MLflow calls are fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from pipelines.train import (
    MIN_F1_THRESHOLD,
    build_model,
    quality_gate,
    train,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARAMS = {
    "n_estimators": 10,
    "learning_rate": 0.1,
    "max_depth": 2,
    "subsample": 1.0,
    "random_state": 42,
}


def _make_splits(n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic dataset large enough to keep stratified split stable."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n).astype(float),
            "MonthlyCharges": rng.uniform(18, 120, n),
            "TotalCharges": rng.uniform(18, 8000, n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n, p=[0.73, 0.27]), name="Churn")
    split = int(n * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ---------------------------------------------------------------------------
# Unit tests — build_model
# ---------------------------------------------------------------------------

def test_build_model_returns_gbc():
    model = build_model(PARAMS)
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == PARAMS["n_estimators"]
    assert model.learning_rate == PARAMS["learning_rate"]
    assert model.max_depth == PARAMS["max_depth"]


# ---------------------------------------------------------------------------
# Unit tests — quality_gate
# ---------------------------------------------------------------------------

def test_quality_gate_passes_at_threshold():
    quality_gate(MIN_F1_THRESHOLD)  # must not raise


def test_quality_gate_passes_above_threshold():
    quality_gate(1.0)  # perfect score — must not raise


def test_quality_gate_fails_below_threshold():
    with pytest.raises(ValueError, match="Quality gate failed"):
        quality_gate(MIN_F1_THRESHOLD - 0.01)


def test_quality_gate_fails_at_zero():
    with pytest.raises(ValueError):
        quality_gate(0.0)


# ---------------------------------------------------------------------------
# Integration tests — train() with mocked MLflow + I/O
# ---------------------------------------------------------------------------

@patch("pipelines.train.mlflow.sklearn.log_model")
@patch("pipelines.train.mlflow.log_artifact")
@patch("pipelines.train.mlflow.log_metrics")
@patch("pipelines.train.mlflow.log_params")
@patch("pipelines.train.mlflow.start_run")
@patch("pipelines.train.mlflow.set_experiment")
@patch("pipelines.train.mlflow.set_tracking_uri")
@patch("pipelines.train.load_splits")
@patch("pipelines.train.load_params", return_value=PARAMS)
def test_train_logs_params_and_metrics(
    mock_load_params,
    mock_load_splits,
    mock_set_uri,
    mock_set_exp,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_artifact,
    mock_log_model,
):
    mock_load_splits.return_value = _make_splits()

    # Use a real context manager for start_run
    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_run_ctx.__exit__ = MagicMock(return_value=False)
    mock_start_run.return_value = mock_run_ctx

    train()

    mock_log_params.assert_called_once_with(PARAMS)

    logged_metrics = mock_log_metrics.call_args[0][0]
    assert set(logged_metrics.keys()) == {"f1", "roc_auc", "precision", "recall"}
    for v in logged_metrics.values():
        assert 0.0 <= v <= 1.0


@patch("pipelines.train.mlflow.sklearn.log_model")
@patch("pipelines.train.mlflow.log_artifact")
@patch("pipelines.train.mlflow.log_metrics")
@patch("pipelines.train.mlflow.log_params")
@patch("pipelines.train.mlflow.start_run")
@patch("pipelines.train.mlflow.set_experiment")
@patch("pipelines.train.mlflow.set_tracking_uri")
@patch("pipelines.train.load_splits")
@patch("pipelines.train.load_params", return_value=PARAMS)
def test_train_logs_two_artifacts(
    mock_load_params,
    mock_load_splits,
    mock_set_uri,
    mock_set_exp,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_artifact,
    mock_log_model,
):
    mock_load_splits.return_value = _make_splits()

    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_run_ctx.__exit__ = MagicMock(return_value=False)
    mock_start_run.return_value = mock_run_ctx

    train()

    assert mock_log_artifact.call_count == 2
    artifact_paths = [call.args[0] for call in mock_log_artifact.call_args_list]
    assert any("confusion_matrix" in p for p in artifact_paths)
    assert any("feature_importance" in p for p in artifact_paths)


@patch("pipelines.train.mlflow.sklearn.log_model")
@patch("pipelines.train.mlflow.log_artifact")
@patch("pipelines.train.mlflow.log_metrics")
@patch("pipelines.train.mlflow.log_params")
@patch("pipelines.train.mlflow.start_run")
@patch("pipelines.train.mlflow.set_experiment")
@patch("pipelines.train.mlflow.set_tracking_uri")
@patch("pipelines.train.load_splits")
@patch("pipelines.train.load_params", return_value=PARAMS)
def test_train_registers_model(
    mock_load_params,
    mock_load_splits,
    mock_set_uri,
    mock_set_exp,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_artifact,
    mock_log_model,
):
    mock_load_splits.return_value = _make_splits()

    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_run_ctx.__exit__ = MagicMock(return_value=False)
    mock_start_run.return_value = mock_run_ctx

    train()

    mock_log_model.assert_called_once()
    call_kwargs = mock_log_model.call_args.kwargs
    assert call_kwargs["registered_model_name"] == "fraud-detector"
    assert call_kwargs["artifact_path"] == "model"


@patch("pipelines.train.mlflow.sklearn.log_model")
@patch("pipelines.train.mlflow.log_artifact")
@patch("pipelines.train.mlflow.log_metrics")
@patch("pipelines.train.mlflow.log_params")
@patch("pipelines.train.mlflow.start_run")
@patch("pipelines.train.mlflow.set_experiment")
@patch("pipelines.train.mlflow.set_tracking_uri")
@patch("pipelines.train.load_splits")
@patch("pipelines.train.load_params", return_value=PARAMS)
def test_train_quality_gate_blocks_registration(
    mock_load_params,
    mock_load_splits,
    mock_set_uri,
    mock_set_exp,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_artifact,
    mock_log_model,
    monkeypatch,
):
    """A degraded dataset that produces low F1 must raise before log_model is called."""
    # All-zero predictions → F1 = 0 → gate must fire
    mock_load_splits.return_value = _make_splits()

    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_run_ctx.__exit__ = MagicMock(return_value=False)
    mock_start_run.return_value = mock_run_ctx

    # Override MIN_F1_THRESHOLD to 1.01 so any real score fails
    monkeypatch.setattr("pipelines.train.MIN_F1_THRESHOLD", 1.01)

    with pytest.raises(ValueError, match="Quality gate failed"):
        train()

    mock_log_model.assert_not_called()
