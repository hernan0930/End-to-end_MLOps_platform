from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)["preprocess"]


def load_data(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Clean, encode, and split the raw Telco Churn dataset.

    Returns X_train, X_test, y_train, y_test as DataFrames/Series.
    """
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges has blank strings in the IBM dataset — convert and impute.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode binary categorical columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)

    # One-hot encode multi-value categorical columns
    present_multi = [c for c in MULTI_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_multi, drop_first=False)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data/processed",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)
    print(f"Saved {len(X_train)} train / {len(X_test)} test samples to {out}")


def main() -> None:
    params = load_params()
    df = load_data("data/raw/dataset.csv")
    X_train, X_test, y_train, y_test = preprocess(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )
    save_splits(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
