import numpy as np
import pandas as pd
import pytest

from pipelines.preprocess import (
    BINARY_COLS,
    MULTI_COLS,
    preprocess,
    save_splits,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal synthetic dataset matching the IBM Telco Churn schema."""
    n = 100
    rng = np.random.default_rng(42)

    total_charges = rng.uniform(18, 8000, n).round(2).astype(str)
    # Inject blank strings to simulate the real dataset's missing values
    total_charges[::10] = " "

    return pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(18, 120, n).round(2),
            "TotalCharges": total_charges,
            "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )


def test_customerid_dropped(raw_df):
    X_train, _, _, _ = preprocess(raw_df.copy())
    assert "customerID" not in X_train.columns


def test_total_charges_no_nulls_after_imputation(raw_df):
    X_train, X_test, _, _ = preprocess(raw_df.copy())
    combined = pd.concat([X_train, X_test])
    assert combined["TotalCharges"].isna().sum() == 0


def test_target_is_binary(raw_df):
    _, _, y_train, y_test = preprocess(raw_df.copy())
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_split_ratio(raw_df):
    n = len(raw_df)
    X_train, X_test, _, _ = preprocess(raw_df.copy(), test_size=0.2)
    assert abs(len(X_test) / n - 0.2) < 0.05


def test_train_test_sizes_sum_to_total(raw_df):
    n = len(raw_df)
    X_train, X_test, y_train, y_test = preprocess(raw_df.copy())
    assert len(X_train) + len(X_test) == n
    assert len(y_train) + len(y_test) == n


def test_multi_cols_are_one_hot_encoded(raw_df):
    X_train, _, _, _ = preprocess(raw_df.copy())
    for col in MULTI_COLS:
        assert col not in X_train.columns, (
            f"Column '{col}' should have been one-hot encoded and removed."
        )


def test_binary_cols_are_numeric(raw_df):
    X_train, _, _, _ = preprocess(raw_df.copy())
    for col in BINARY_COLS:
        if col in X_train.columns:
            assert X_train[col].dtype in [int, float, "int64", "float64"], (
                f"Column '{col}' should be numeric after binary encoding."
            )


def test_no_object_columns_remain(raw_df):
    X_train, _, _, _ = preprocess(raw_df.copy())
    object_cols = X_train.select_dtypes(include="object").columns.tolist()
    assert object_cols == [], f"Unexpected object columns remaining: {object_cols}"


def test_save_splits_creates_four_files(raw_df, tmp_path):
    X_train, X_test, y_train, y_test = preprocess(raw_df.copy())
    save_splits(X_train, X_test, y_train, y_test, output_dir=str(tmp_path))
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert (tmp_path / fname).exists(), f"Expected file not found: {fname}"


def test_save_splits_row_counts_match(raw_df, tmp_path):
    X_train, X_test, y_train, y_test = preprocess(raw_df.copy())
    save_splits(X_train, X_test, y_train, y_test, output_dir=str(tmp_path))
    assert len(pd.read_csv(tmp_path / "X_train.csv")) == len(X_train)
    assert len(pd.read_csv(tmp_path / "X_test.csv")) == len(X_test)
    assert len(pd.read_csv(tmp_path / "y_train.csv")) == len(y_train)
    assert len(pd.read_csv(tmp_path / "y_test.csv")) == len(y_test)
