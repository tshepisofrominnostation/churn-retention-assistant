"""Feature pipeline and model definition for the churn classifier."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "Age", "Tenure_Months", "Balance", "Num_Products",
    "Estimated_Salary", "Transaction_Count_30D", "Support_Tickets_30D",
]
BINARY_FEATURES = ["Has_Credit_Card", "Is_Active_Member"]
FEATURE_ORDER = NUMERIC_FEATURES + BINARY_FEATURES          # input order for predict
RISK_BANDS = [(0.30, "Low"), (0.70, "Medium"), (1.01, "High")]


def build_pipeline() -> Pipeline:
    """Return the full preprocessing + classifier pipeline.

    Scaling is included even though tree models do not require it: it keeps
    the pipeline consistent for future model swaps (logistic regression, SVM).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )
    # Model selection (see README): a Random Forest reached ROC-AUC 0.81 but
    # its probabilities compressed to a max of ~0.68, never crossing the 70%
    # high-risk band even for worst-case profiles. Gradient boosted trees keep
    # probabilities well separated (worst-case ~0.9, engaged ~0.1) at equal AUC,
    # so risk banding is meaningful.
    model = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05, random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])
