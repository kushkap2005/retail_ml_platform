# 🛒 Retail ML Analytics Platform

An end-to-end retail intelligence platform covering three ML modules deployed on a unified Streamlit dashboard.

---

## 📦 Project Structure

```
retail_ml_platform/
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── sales_data.csv            # 10,000 rows · 6 categories · 25 SKUs · 4 channels
│   └── customer_data.csv         # 2,000 customer profiles
└── models/
    ├── demand_forecast.py        # Module 1: LSTM demand forecasting
    ├── churn_prediction.py       # Module 2: XGBoost + SHAP churn prediction
    └── price_elasticity.py       # Module 3: EconML causal price elasticity
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data (if CSVs not present)
```bash
python data/generate_data.py
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🤖 ML Modules

### Module 1 — Demand Forecasting (LSTM)
- **Model**: LSTM neural network (TensorFlow/Keras) with 14-day sliding window
- **Fallback**: Moving average when TF unavailable or insufficient data
- **Features**: Daily aggregated units sold, seasonality, channel effects
- **Outputs**: 7–60 day demand forecast with confidence band, MAE / RMSE / MAPE
- **Drill-down**: Filter by category, channel, or individual SKU

### Module 2 — Customer Churn Prediction (XGBoost + SHAP)
- **Model**: XGBoost classifier (200 trees, depth 4)
- **Explainability**: SHAP TreeExplainer — global feature importance + individual prediction waterfall
- **Features**: Tenure, purchase frequency, RFM signals, promo response, channel diversity
- **Outputs**: Accuracy, AUC-ROC, confusion matrix, ROC curve, churn probability histogram
- **Interactive**: Single-customer churn risk predictor with sliders

### Module 3 — Price Elasticity (EconML)
- **Model**: Double ML (LinearDML) with GradientBoosting nuisance models
- **Fallback**: OLS log-log regression when < 60 samples
- **Controls for**: promotions, marketing spend, seasonality, inventory
- **Outputs**: Elasticity coefficient + 95% CI per category/channel
- **Simulator**: Waterfall revenue impact for any % price change

---

## 📊 Dataset

| Attribute       | Value              |
|-----------------|--------------------|
| Total rows      | 10,000 (sales)     |
| Customer rows   | 2,000              |
| Date range      | Jan 2022 – Dec 2023 |
| Categories      | 6                  |
| SKUs            | 25                 |
| Channels        | 4                  |

**Sales features**: date, sku, category, channel, price, discount_rate, units_sold, revenue, cogs, profit, marketing_spend, promotion_flag, inventory_level, day_of_week, month, quarter, is_weekend

**Customer features**: tenure_months, purchase_frequency, avg_order_value, days_since_last_purchase, category_diversity, channel_diversity, promo_response_rate, email_open_rate, support_calls, churned

---

## 🗂️ JD Coverage

| JD Project | Coverage |
|---|---|
| P1 · Supply Chain Cost Optimization | LSTM forecasting + EconML elasticity |
| P2 · Customer Forecasting | LSTM + XGBoost demand/churn models |
| P3 · Customer Lifecycle / NBA | XGBoost churn + SHAP attribution |
| P4 · Supply Chain Forecasting | LSTM at SKU level across channels |
| P5 · Agentic AI | Not covered (separate project needed) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit + Plotly |
| Forecasting | TensorFlow/Keras LSTM |
| Classification | XGBoost |
| Explainability | SHAP TreeExplainer |
| Causal ML | EconML (LinearDML) |
| Data | Pandas, NumPy, Scikit-learn |
