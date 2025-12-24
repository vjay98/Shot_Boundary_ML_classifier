from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

# ---------- USER CONFIG ----------
CSV_DIR = "event_features"
OUT_DIR = "models_shotcls"
CLASSES_TO_KEEP = None
DEFAULT_FEATURES = [
    "max_hist",
    "width_hist",
    "pre_edge",
    "post_edge",
    "pre_frac",
    "post_frac",
    "pre_hist",
    "post_hist",
    "phash",
    "cx_trend",
]
# ---------------------------------


def load_all_event_features(csv_dir: str) -> pd.DataFrame:
    csv_dir = Path(csv_dir)
    files = list(csv_dir.glob("event_features_*.csv"))
    if not files:
        raise FileNotFoundError(f"No event_features_*.csv in {csv_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["stream_id"] = f.stem.replace("event_features_", "")
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {f.name}")
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_df)}")
    return all_df


def prepare_dataset(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    df = df.copy()
    df["edge_ratio_post_pre"] = df["post_edge"] / (df["pre_edge"] + 1e-6)
    df["edge_ratio_pre_post"] = df["pre_edge"] / (df["post_edge"] + 1e-6)
    df["frac_mean"] = (df["pre_frac"] + df["post_frac"]) / 2.0

    feature_cols = feature_cols + [
        "edge_ratio_post_pre",
        "edge_ratio_pre_post",
        "frac_mean",
    ]

    if CLASSES_TO_KEEP is not None:
        df = df[df["kind"].isin(CLASSES_TO_KEEP)].reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y = df["kind"].values

    # --- handle NaNs: median per feature ---
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    return X, y, feature_cols, imputer


def train_models(X, y, out_dir: str, feature_cols, imputer: SimpleImputer):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_lr, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred_lr))

    joblib.dump(
        {"model": lr, "scaler": scaler, "imputer": imputer, "features": feature_cols},
        out_dir / "shot_lr_model.joblib",
    )
    print(f"Saved LR model to {out_dir / 'shot_lr_model.joblib'}")

    # Decision Tree (can work on unscaled X)
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

    print("\n=== Decision Tree (depth=5) ===")
    print(classification_report(y_test, y_pred_tree, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred_tree))

    joblib.dump(
        {"model": tree, "imputer": imputer, "features": feature_cols},
        out_dir / "shot_tree_model.joblib",
    )
    print(f"Saved tree model to {out_dir / 'shot_tree_model.joblib'}")


if __name__ == "__main__":
    df = load_all_event_features(CSV_DIR)
    X, y, feat_cols, imputer = prepare_dataset(df)
    print(f"Using {len(feat_cols)} features: {feat_cols}")
    train_models(X, y, OUT_DIR, feat_cols, imputer)
