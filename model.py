
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)

TARGET_COLUMN = "status"

FEATURE_COLUMNS = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


def prepare_dataset(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Dataset must contain a 'status' column."
        )

    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    if not available_features:
        raise ValueError(
            "No expected Parkinson's feature columns found."
        )

    X = df[available_features].apply(
        pd.to_numeric, errors="coerce"
    )

    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    # Remove rows where every feature is missing.
    valid_features = X.notna().any(axis=1)
    X = X.loc[valid_features].copy()
    y = y.loc[valid_features].copy()

    return X, y, available_features


class FuzzyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="distance",
            )),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def evaluate_models(X, y):
    results = {}

    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if len(y) < 4 or y.nunique() < 2:
        raise ValueError(
            "Not enough valid data or only one target class found."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Fuzzy KNN": FuzzyKNN(n_neighbors=5),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                probability=True,
                random_state=42,
            )),
        ]),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "Accuracy": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "F1 Score": round(
                f1_score(
                    y_test, predictions, zero_division=0
                ), 4
            ),
            "MCC": round(
                matthews_corrcoef(y_test, predictions), 4
            ),
        }

    # Train the final Fuzzy KNN model using the complete dataset.
    final_model = FuzzyKNN(n_neighbors=5)
    final_model.fit(X, y)

    # These objects are returned for compatibility with app.py.
    final_scaler = final_model.pipeline.named_steps["scaler"]

    return results, final_model, final_scaler, X, y
