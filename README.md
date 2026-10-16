# Customer Churn Predictor & Retention Assistant

A production-style fintech prototype that predicts retail-banking customer churn with machine learning and hands the high-risk cases to a generative AI retention assistant that drafts a personalised, compliance-checked retention email.

**Author:** Freddy Thosago · Python · scikit-learn · Streamlit · LLM automation

---

## Problem statement

Banks typically know a customer has left only after the account is closed. This project moves that moment earlier: behavioural signals (activity, support friction, tenure, balance, product intensity) are scored by a classification model, and customers crossing a **70% churn-risk band** are automatically queued for an empathetic, compliant retention offer drafted by an LLM - with a deterministic template fallback and a hard guardrail against hallucinated financial promises.

## Architecture

```
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  Data Input  │───▶│   ML Classifier    │───▶│ Risk Assessment      │
│  (CSV / UI)  │    │ Gradient Boosted   │    │ Low <30% / Med 30-70%│
│              │    │ Trees + scaler     │    │ High >70% ───────────┼──┐
└──────────────┘    └────────────────────┘    └──────────────────────┘  │
                                                                        ▼
              ┌───────────────────────────────────────────────────────────────┐
              │  Generative AI Retention Assistant (LLM, env-keyed)          │
              │  profile + inferred drivers + APPROVED-OFFER whitelist       │
              │  ──▶ personalised email ──▶ compliance validator ──▶ output  │
              └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │ Streamlit   │
                                   │ Dashboard   │
                                   └─────────────┘
```

**How the layers connect:** the ML layer outputs a churn *probability*; the risk engine converts it to a band; only `High` customers flow to the generative layer. The LLM never sees raw PII (no names/account numbers - the prompt receives behaviour metrics only) and can only reference offers from a whitelist. Its output is re-checked by a compliance validator that blocks guarantee-style language.

## Results (held-out test set, 100 customers)

| Metric | Value |
|---|---|
| ROC-AUC | **0.800** |
| Precision | 0.583 |
| Recall | 0.438 |
| F1 | 0.500 |
| Dataset | 500 synthetic customers, 31.6% churn |
| Top drivers (permutation importance) | `Is_Active_Member`, `Tenure_Months`, `Support_Tickets_30D`, `Transaction_Count_30D` |

**Model selection note:** a Random Forest reached ROC-AUC 0.81 but its class-weighted, calibrated probabilities compressed to a maximum of ~0.68 - never crossing the 70% high-risk band even for worst-case profiles (worst-case test profile scored 55%). Gradient boosted trees (`HistGradientBoostingClassifier`) matched the AUC while keeping probabilities well separated: worst-case profile **89%**, high-friction profile **81%**, engaged loyal customer **8%**. Banded risk output requires that separation, so HGB was chosen.

Sanity-check profiles (trained model):

| Profile | Churn probability | Band |
|---|---|---|
| 8 support tickets, inactive, 2-month tenure | 88.7% | High |
| 5 tickets, inactive, single product | 81.2% | High |
| Active, 30-month tenure, healthy usage | 15.3% | Low |
| Active, 96-month tenure, 3 products, 18 tx/month | 7.9% | Low |

## Repository layout

```
churn-retention-assistant/
├── app.py                        # Streamlit UI (sidebar → gauge → AI assistant)
├── data/
│   ├── generate_dataset.py       # seeded synthetic data generator
│   └── churn_dataset.csv         # 500 synthetic customers
├── src/
│   ├── features.py               # pipeline definition (scaling + model)
│   ├── train.py                  # train, evaluate, persist model + metrics.json
│   ├── predict.py                # load model, score profiles, risk bands
│   └── retention_assistant.py    # LLM drafting + fallback + compliance guardrail
├── models/                       # churn_model.joblib (gitignored), metrics.json
├── docs/Responsible_AI.md        # ethical risks and engineering guardrails
├── requirements.txt              # pinned
├── .env.example                  # LLM API key config (never hardcoded)
└── README.md
```

## Setup

```bash
git clone https://github.com/tshepisofrominnostation/churn-retention-assistant.git
cd churn-retention-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate data (already bundled, regenerates deterministically)
python data/generate_dataset.py

# 2. Train the model (writes models/churn_model.joblib + metrics.json)
python -m src.train

# 3. Launch the dashboard
streamlit run app.py
```

Optional: copy `.env.example` to `.env` and set `LLM_API_KEY` (any OpenAI-compatible endpoint) to switch the retention assistant from template fallback to live LLM drafting. The app trains automatically on first run if no model file exists.

## Responsible AI (summary - full write-up in `docs/Responsible_AI.md`)

1. **Socio-economic bias:** salary/balance proxies can steer offers away from low-income customers. Guardrails: permutation-importance review, fairness monitoring by salary quartile and age band, human approval before offer dispatch, no protected attributes as features.
2. **Privacy / POPIA:** customer PII must never reach a third-party LLM. Guardrails: prompt receives behavioural aggregates only ("Valued customer"), data minimisation, env-keyed access, on-prem LLM option for production.
3. **Hallucinated financial promises:** LLMs can invent guaranteed approvals or rates. Guardrails: approved-offer whitelist in the prompt, post-generation compliance validator (`check_compliance`) that blocks guarantee language, deterministic template fallback, human-in-the-loop review before sending.

## Future enhancements

- Threshold optimisation on a validation split (recall-weighted PR curve) to lift the 0.438 recall
- SHAP-based per-customer explanations ("why this customer is high risk")
- Batch scoring endpoint + database persistence for campaign management
- A/B testing harness to measure offer acceptance vs. control
- Drift monitoring on feature distributions (banking behaviour shifts fast)
