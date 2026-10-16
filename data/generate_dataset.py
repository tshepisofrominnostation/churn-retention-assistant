"""Generate a realistic synthetic retail-banking churn dataset (~500 records).

The dataset is SYNTHETIC: customer attributes are sampled from realistic
distributions, and churn is drawn from a logistic model of behaviour drivers
(activity, support load, tenure, balance, product intensity) plus random
noise. No real customers are involved.

Run:  python data/generate_dataset.py  ->  data/churn_dataset.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 500


def generate_dataset(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """Return a synthetic banking dataset with a behaviour-driven churn label.

    Args:
        n: Number of customer records to generate.
        seed: Random seed for full reproducibility.

    Returns:
        DataFrame with demographic, behavioural and churn columns.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(38, 12, n), 18, 85).astype(int)
    tenure_months = rng.integers(1, 121, n)                      # 1 month - 10 years
    balance = np.round(np.clip(rng.lognormal(9.0, 1.3, n), 0, 2_500_000), 2)
    num_products = rng.integers(1, 5, n)                         # 1-4 products
    has_credit_card = rng.binomial(1, 0.62, n)
    is_active_member = rng.binomial(1, 0.55, n)
    estimated_salary = np.round(np.clip(rng.normal(28_000, 14_000, n), 5_000, 250_000), 2)
    transaction_count_30d = np.where(
        is_active_member == 1,
        rng.poisson(14, n),                                     # active: ~14 tx/month
        rng.poisson(3, n),                                      # inactive: ~3 tx/month
    )
    support_tickets_30d = rng.poisson(0.5, n) + np.where(
        num_products >= 3, rng.poisson(0.6, n), 0               # more products, more issues
    )

    # ── Churn drivers (logistic model) ─────────────────────────────────────
    # Inactivity, high support load, short tenure and low balance all raise risk.
    z = (
        1.00                                             # intercept, calibrated for ~25% churn
        - 1.60 * is_active_member                        # active members rarely churn
        + 0.28 * support_tickets_30d                     # frustration signal
        - 0.018 * tenure_months                          # loyalty effect
        - 0.35 * np.log1p(balance) / 10                  # bigger balance, stickier
        - 0.10 * transaction_count_30d / 10              # usage stickiness
        + 0.22 * (num_products == 1).astype(int)         # single-product customers drift
        + 0.012 * (age < 30).astype(int) * 10            # young switchers
        + rng.normal(0, 0.35, n)                          # unobserved noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "Customer_ID": [f"CUST{1000 + i}" for i in range(n)],
        "Age": age,
        "Tenure_Months": tenure_months,
        "Balance": balance,
        "Num_Products": num_products,
        "Has_Credit_Card": has_credit_card,
        "Is_Active_Member": is_active_member,
        "Estimated_Salary": estimated_salary,
        "Transaction_Count_30D": transaction_count_30d,
        "Support_Tickets_30D": support_tickets_30d,
        "Churn": churn,
    })
    return df


def main() -> None:
    """Write the dataset to data/churn_dataset.csv and print a summary."""
    out = Path(__file__).parent / "churn_dataset.csv"
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
