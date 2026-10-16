"""Inference helpers: load the trained pipeline and score customer profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.features import FEATURE_ORDER, RISK_BANDS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
_PIPELINE = None


def load_model():
    """Load and cache the trained churn pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "models/churn_model.joblib not found - run `python -m src.train` first."
            )
        _PIPELINE = joblib.load(MODEL_PATH)
    return _PIPELINE


def risk_level(probability: float) -> str:
    """Map a churn probability to a Low / Medium / High band."""
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return "High"


def predict_churn(profile: Dict[str, float]) -> Dict[str, object]:
    """Score one customer profile and return probability, risk level and band.

    Args:
        profile: Dict with the ten model features (see FEATURE_ORDER).

    Returns:
        {"churn_probability": 0-1 float, "churn_probability_pct": str,
         "risk_level": "Low" | "Medium" | "High"}
    """
    missing = [f for f in FEATURE_ORDER if f not in profile]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model = load_model()
    row = pd.DataFrame([[profile[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "churn_probability_pct": f"{probability:.1%}",
        "risk_level": risk_level(probability),
    }
