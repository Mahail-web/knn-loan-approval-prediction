"""
Data Preprocessing Pipeline for Loan Approval Prediction using KNN.
Provides column definitions, transformer pipelines, and data loading utilities.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definitions
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "Loan_Status"

# Target mapping: Y -> 1 (Approved), N -> 0 (Rejected)
TARGET_MAPPING = {"N": 0, "Y": 1}
TARGET_NAMES = {0: "Rejected", 1: "Approved"}


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Remove Loan_ID, validate columns, and separate features (X) and target (y).
    """
    clean_df = df.copy()
    if "Loan_ID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["Loan_ID"])

    X = clean_df[ALL_FEATURES].copy()
    y = clean_df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical and categorical preprocessing:
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train-test split (80% train, 20% test).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

