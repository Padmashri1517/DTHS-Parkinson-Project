import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "status"

def prepare_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column 'status' was not found. Please use the UCI Parkinson's dataset or adapt TARGET_COLUMN.")
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    # Remove target and common identifier fields so they cannot leak into the model.
    drop_cols = [TARGET_COLUMN]
    for col in ["name", "subject", "patient_id", "id"]:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    # Keep numerical columns with at least one numeric value.
    X = X.dropna(axis=1, how="all")
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numerical feature columns were found.")
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    return X, y, list(X.columns)

class FuzzyKNN:
    """Simple fuzzy k-NN: inverse-distance neighbour weights are summed per class."""
    def __init__(self, n_neighbors=5, power=2.0, eps=1e-9):
        self.n_neighbors = int(n_neighbors)
        self.power = float(power)
        self.eps = float(eps)

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        if len(self.X_) < self.n_neighbors:
            self.n_neighbors = max(1, len(self.X_))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        all_probs = []
        for row in X:
            distances = np.sqrt(np.sum((self.X_ - row) ** 2, axis=1))
            idx = np.argsort(distances)[:self.n_neighbors]
            d = distances[idx]
            labels = self.y_[idx]
            # Exact match: give all mass to the matching neighbour class(es).
            if np.any(d < self.eps):
                exact_labels = labels[d < self.eps]
                probs = np.array([(exact_labels == c).mean() for c in self.classes_], dtype=float)
            else:
                weights = 1.0 / np.maximum(d, self.eps) ** self.power
                probs = np.array([weights[labels == c].sum() for c in self.classes_], dtype=float)
                probs = probs / probs.sum()
            all_probs.append(probs)
        return np.asarray(all_probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

def evaluate_models(X, y):
    # Stratify preserves class proportions where possible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    # Tune k on training data only. This is a practical student-project version,
    # not a reproduction of the paper's full PSO-based joint feature/k/m optimization.
    min_class = int(y_train.value_counts().min())
    n_splits = min(3, min_class)
    if n_splits < 2:
        raise ValueError("Not enough examples per class for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    k_values = [k for k in [3, 5, 7, 9, 11] if k < len(X_train_s)]
    if not k_values:
        k_values = [1]
    knn_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    grid = GridSearchCV(
        knn_pipe,
        {"knn__n_neighbors": k_values, "knn__weights": ["uniform", "distance"]},
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    standard_knn = grid.best_estimator_
    knn_pred = standard_knn.predict(X_test)

    fuzzy = FuzzyKNN(n_neighbors=min(grid.best_params_["knn__n_neighbors"], len(X_train_s)-1), power=2)
    fuzzy.fit(X_train_s, y_train.to_numpy())
    fuzzy_pred = fuzzy.predict(X_test_s)

    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    rows = []
    for name, pred in [
        ("Fuzzy k-NN (student prototype)", fuzzy_pred),
        ("Tuned standard KNN", knn_pred),
        ("RBF SVM baseline", svm_pred),
    ]:
        rows.append({
            "Model": name,
            "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "F1-score": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "MCC": round(float(matthews_corrcoef(y_test, pred)), 4),
        })
    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Refit imputer/scaler on all usable data for the demo predictor.
    X_all_i = imputer.fit_transform(X)
    scaler.fit(X_all_i)
    X_all_s = scaler.transform(X_all_i)
    fuzzy.fit(X_all_s, y.to_numpy())
    # For the demo, return a scaler that includes imputation in a simple wrapper-compatible way.
    class ImputeScale:
        def __init__(self, imp, sc):
            self.imp, self.sc = imp, sc
        def transform(self, data):
            return self.sc.transform(self.imp.transform(data))
    scaler_bundle = ImputeScale(imputer, scaler)
    X_clean = pd.DataFrame(X_all_i, columns=X.columns)
    return results, fuzzy, scaler_bundle, X_clean, y.reset_index(drop=True)
