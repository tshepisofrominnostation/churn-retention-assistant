"""Train, evaluate and persist the churn classifier.

Run:  python -m src.train   (from the repo root)
Writes:  models/churn_model.joblib, models/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_ORDER, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "churn_dataset.csv"
MODEL_DIR = ROOT / "models"


def load_data() -> pd.DataFrame:
    """Load the synthetic churn dataset from the CSV."""
    return pd.read_csv(DATA_PATH)


def train() -> dict:
    """Train the pipeline, print real evaluation metrics and persist artifacts."""
    df = load_data()
    X, y = df[FEATURE_ORDER], df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 3),
        "precision": round(float(precision_score(y_test, y_pred)), 3),
        "recall": round(float(recall_score(y_test, y_pred)), 3),
        "f1": round(float(f1_score(y_test, y_pred)), 3),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    # Permutation importance: model-agnostic (HGB has no feature_importances_).
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(pipeline, X_test, y_test,
                                  n_repeats=10, random_state=42, n_jobs=-1)
    importances = perm.importances_mean

    ranked = sorted(
        zip(FEATURE_ORDER, (round(float(v), 3) for v in importances)),
        key=lambda kv: -kv[1],
    )
    metrics["top_features"] = ranked[:5]

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "churn_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
