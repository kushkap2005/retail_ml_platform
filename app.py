"""
Retail ML Platform — Streamlit Dashboard
Modules: Demand Forecasting (LSTM) | Churn Prediction (XGBoost+SHAP) | Price Elasticity (EconML)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config 
st.set_page_config(
    page_title="Retail ML Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .metric-card {
        background: white; border-radius: 12px;
        padding: 1rem 1.4rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {background-color: #4c78a8; color: white;}
    h1 {color: #1a1a2e;}
    h2,h3 {color: #2d3561;}
    div[data-testid="stSidebarContent"] {background-color: #1a1a2e;}
    div[data-testid="stSidebarContent"] * {color: #e8eaf6 !important;}
</style>
""", unsafe_allow_html=True)


# ── Data loading 
@st.cache_data(show_spinner=False)
def load_data():
    base = os.path.dirname(__file__)
    sales_path = os.path.join(base, "data", "sales_data.csv")
    cust_path  = os.path.join(base, "data", "customer_data.csv")
    if not os.path.exists(sales_path):
        from data.generate_data import generate_sales_data, generate_customer_data
        os.makedirs(os.path.join(base, "data"), exist_ok=True)
        sales = generate_sales_data(10000)
        cust  = generate_customer_data(2000)
        sales.to_csv(sales_path, index=False)
        cust.to_csv(cust_path, index=False)
    else:
        sales = pd.read_csv(sales_path, parse_dates=["date"])
        cust  = pd.read_csv(cust_path)
    return sales, cust


# ── Model caching 
@st.cache_resource(show_spinner=False)
def get_churn_model(cust_df):
    from models.churn_prediction import train_churn_model
    return train_churn_model(cust_df)

@st.cache_data(show_spinner=False)
def get_elasticity(sales_df, group_by):
    from models.price_elasticity import compute_elasticity
    return compute_elasticity(sales_df, group_by=group_by)

@st.cache_data(show_spinner=False)
def get_forecast(sales_df_json, sku, category, channel, horizon):
    sales_df = pd.read_json(sales_df_json, orient="records")
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    from models.demand_forecast import train_forecast
    return train_forecast(
        sales_df,
        sku=sku if sku != "All" else None,
        category=category if category != "All" else None,
        channel=channel if channel != "All" else None,
        horizon=horizon,
    )


# ── Helpers 
PALETTE = px.colors.qualitative.Set2


def kpi_row(metrics: list):
    """metrics = list of (label, value, delta, delta_color)"""
    cols = st.columns(len(metrics))
    for col, (label, value, delta, color) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta=delta, delta_color=color)


#  SIDEBAR

with st.sidebar:
    st.markdown("## 🛒 Retail ML Platform")
    st.markdown("---")
    st.markdown("### Global Filters")

    sales_df_raw, cust_df_raw = load_data()

    categories = ["All"] + sorted(sales_df_raw["category"].unique().tolist())
    channels   = ["All"] + sorted(sales_df_raw["channel"].unique().tolist())
    skus       = ["All"] + sorted(sales_df_raw["sku"].unique().tolist())

    sel_category = st.selectbox("Category", categories)
    sel_channel  = st.selectbox("Channel",  channels)

    st.markdown("---")
    st.markdown("### Date Range")
    min_date = sales_df_raw["date"].min().date()
    max_date = sales_df_raw["date"].max().date()
    date_range = st.date_input("Select range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.caption("📊 Dataset: 10,000 rows · 6 categories · 25 SKUs · 4 channels")
    st.caption("🤖 Models: LSTM · XGBoost · EconML")


# ── Apply global filters 
sales_df = sales_df_raw.copy()
if len(date_range) == 2:
    sales_df = sales_df[
        (sales_df["date"].dt.date >= date_range[0]) &
        (sales_df["date"].dt.date <= date_range[1])
    ]
if sel_category != "All":
    sales_df = sales_df[sales_df["category"] == sel_category]
if sel_channel != "All":
    sales_df = sales_df[sales_df["channel"] == sel_channel]


#  HEADER

st.title("🛒 Retail ML Analytics Platform")
st.markdown(
    "**End-to-end retail intelligence** · Demand Forecasting · Churn Prediction · Price Elasticity"
)
st.markdown("---")

# ── Summary KPIs 
total_rev    = sales_df["revenue"].sum()
total_units  = sales_df["units_sold"].sum()
avg_price    = sales_df["price"].mean()
total_profit = sales_df["profit"].sum()
margin_pct   = total_profit / total_rev * 100 if total_rev > 0 else 0

kpi_row([
    ("Total Revenue",  f"${total_rev:,.0f}",    None, "normal"),
    ("Units Sold",     f"{total_units:,}",       None, "normal"),
    ("Avg Price",      f"${avg_price:.2f}",      None, "normal"),
    ("Total Profit",   f"${total_profit:,.0f}",  None, "normal"),
    ("Profit Margin",  f"{margin_pct:.1f}%",     None, "normal"),
])
st.markdown("---")



#  TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", "🔮 Demand Forecasting", "👥 Churn Prediction", "💰 Price Elasticity"
])


# ─── TAB 1: OVERVIEW 
with tab1:
    st.subheader("Business Overview")

    col1, col2 = st.columns(2)
    with col1:
        # Revenue by category
        rev_cat = sales_df.groupby("category")["revenue"].sum().reset_index()
        fig = px.bar(rev_cat, x="category", y="revenue", color="category",
                     title="Revenue by Category", color_discrete_sequence=PALETTE,
                     labels={"revenue": "Revenue ($)", "category": "Category"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue by channel
        rev_ch = sales_df.groupby("channel")["revenue"].sum().reset_index()
        fig2 = px.pie(rev_ch, values="revenue", names="channel",
                      title="Revenue Share by Channel",
                      color_discrete_sequence=PALETTE, hole=0.35)
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Revenue trend
    trend = (sales_df.groupby(sales_df["date"].dt.to_period("W").dt.start_time)["revenue"]
             .sum().reset_index())
    trend.columns = ["week", "revenue"]
    fig3 = px.area(trend, x="week", y="revenue", title="Weekly Revenue Trend",
                   color_discrete_sequence=["#4c78a8"],
                   labels={"revenue": "Revenue ($)", "week": "Week"})
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Units by SKU top 10
        sku_units = (sales_df.groupby("sku")["units_sold"].sum()
                     .nlargest(10).reset_index())
        fig4 = px.bar(sku_units, x="units_sold", y="sku", orientation="h",
                      title="Top 10 SKUs by Units Sold",
                      color="units_sold", color_continuous_scale="Blues")
        fig4.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        # Discount vs units scatter
        sample = sales_df.sample(min(500, len(sales_df)), random_state=42)
        fig5 = px.scatter(sample, x="discount_rate", y="units_sold",
                          color="category", title="Discount Rate vs Units Sold",
                          color_discrete_sequence=PALETTE, opacity=0.6,
                          labels={"discount_rate": "Discount Rate",
                                  "units_sold": "Units Sold"})
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)


# ─── TAB 2: DEMAND FORECASTING 
with tab2:
    st.subheader("🔮 Demand Forecasting — LSTM Neural Network")
    st.info("LSTM model trained on historical daily demand per category/SKU/channel. "
            "Scroll down to see forecast and performance metrics.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        fc_category = st.selectbox("Category", categories, key="fc_cat")
    with c2:
        fc_channel  = st.selectbox("Channel", channels, key="fc_ch")
    with c3:
        fc_sku      = st.selectbox("SKU (overrides category)", ["All"] + sorted(sales_df_raw["sku"].unique()), key="fc_sku")
    with c4:
        fc_horizon  = st.slider("Forecast days", 7, 60, 30)

    if st.button("▶ Run Forecast", type="primary"):
        with st.spinner("Training LSTM model…"):
            result = get_forecast(
                sales_df_raw.to_json(orient="records"),
                sku=fc_sku if fc_sku != "All" else None,
                category=fc_category if fc_category != "All" else None,
                channel=fc_channel if fc_channel != "All" else None,
                horizon=fc_horizon,
            )
        st.session_state["fc_result"] = result

    if "fc_result" in st.session_state:
        r = st.session_state["fc_result"]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Model", r["model_type"].split("(")[0].strip())
        col_m2.metric("MAE", f"{r['mae']:.2f} units")
        col_m3.metric("RMSE", f"{r['rmse']:.2f} units")
        col_m4.metric("MAPE", f"{r['mape']:.1f}%")

        # Plot
        hist_dates   = pd.to_datetime(r["dates_hist"])
        future_dates = pd.to_datetime(r["dates_future"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_dates, y=r["actual"],
            name="Historical demand", line=dict(color="#4c78a8", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=r["forecast"],
            name="Forecast", line=dict(color="#e45756", width=2, dash="dash")
        ))
        # Confidence band (±10% heuristic)
        fc_arr = np.array(r["forecast"])
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
            y=np.concatenate([fc_arr * 1.10, fc_arr[::-1] * 0.90]),
            fill="toself", fillcolor="rgba(228,87,86,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence band", showlegend=True
        ))
        fig.update_layout(
            title=f"Demand Forecast — {fc_horizon}-Day Horizon",
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", y=1.02), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        fc_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Forecast Units": [round(v, 1) for v in r["forecast"]],
        })
        with st.expander("📋 Forecast table"):
            st.dataframe(fc_df, use_container_width=True, height=300)
    else:
        st.info("👆 Configure filters above and click **Run Forecast** to train the LSTM model.")


# ─── TAB 3: CHURN PREDICTION 
with tab3:
    st.subheader("👥 Customer Churn Prediction — XGBoost + SHAP")
    st.info("XGBoost classifier trained on 2,000 customer profiles. "
            "SHAP values explain individual predictions.")

    with st.spinner("Training churn model…"):
        churn_result = get_churn_model(cust_df_raw)

    # KPIs
    kpi_row([
        ("Accuracy",     f"{churn_result['accuracy']*100:.1f}%",  None, "normal"),
        ("AUC-ROC",      f"{churn_result['auc']:.4f}",             None, "normal"),
        ("Churn Rate",   f"{cust_df_raw['churned'].mean()*100:.1f}%", None, "normal"),
        ("Customers",    f"{len(cust_df_raw):,}",                  None, "normal"),
    ])

    col1, col2 = st.columns(2)
    with col1:
        # Feature importance (SHAP)
        fi = churn_result["feature_importance"]
        fig_fi = px.bar(fi, x="mean_abs_shap", y="feature", orientation="h",
                        title="Feature Importance (Mean |SHAP|)",
                        color="mean_abs_shap", color_continuous_scale="Blues",
                        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""})
        fig_fi.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        # Confusion matrix
        cm = np.array(churn_result["confusion_matrix"])
        labels = ["Retained", "Churned"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(churn_result["y_test"], churn_result["y_prob"])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"AUC = {churn_result['auc']:.4f}",
                                  line=dict(color="#4c78a8", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  name="Random", line=dict(color="gray", dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", height=320)
    st.plotly_chart(fig_roc, use_container_width=True)

    # Churn probability distribution
    pred_df = churn_result["predictions_df"]
    fig_hist = px.histogram(pred_df, x="churn_prob", color="churned",
                             nbins=40, barmode="overlay",
                             color_discrete_map={0: "#54a24b", 1: "#e45756"},
                             title="Churn Probability Distribution",
                             labels={"churn_prob": "Predicted Churn Probability",
                                     "churned": "Actual Churn"})
    fig_hist.update_layout(height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Individual prediction
    st.subheader("🔍 Individual Customer Predictor")
    with st.expander("Try a custom customer profile"):
        i1, i2, i3 = st.columns(3)
        with i1:
            tenure       = st.slider("Tenure (months)", 1, 36, 12)
            freq         = st.slider("Purchase frequency", 0, 20, 3)
            avg_spend    = st.slider("Avg order value ($)", 10, 500, 100)
        with i2:
            days_last    = st.slider("Days since last purchase", 1, 180, 45)
            cat_div      = st.slider("Category diversity", 1, 6, 3)
            ch_div       = st.slider("Channel diversity", 1, 4, 2)
        with i3:
            promo_resp   = st.slider("Promo response rate", 0.0, 1.0, 0.4)
            email_open   = st.slider("Email open rate", 0.0, 1.0, 0.3)
            support      = st.slider("Support calls", 0, 10, 1)

        if st.button("Predict Churn Risk"):
            from models.churn_prediction import predict_churn
            customer = {
                "tenure_months": tenure, "purchase_frequency": freq,
                "avg_order_value": avg_spend, "days_since_last_purchase": days_last,
                "category_diversity": cat_div, "channel_diversity": ch_div,
                "promo_response_rate": promo_resp, "email_open_rate": email_open,
                "support_calls": support,
            }
            res = predict_churn(churn_result, customer)
            prob = res["churn_probability"]
            risk_label = "🔴 High Risk" if prob > 0.7 else ("🟡 Medium Risk" if prob > 0.4 else "🟢 Low Risk")
            st.metric("Churn Probability", f"{prob:.1%}", delta=risk_label, delta_color="off")


# ─── TAB 4: PRICE ELASTICITY 
with tab4:
    st.subheader("💰 Price Elasticity Modeling — EconML Causal ML")
    st.info("Double ML (LinearDML) estimates causal price elasticity, controlling for "
            "promotions, marketing, and seasonality.")

    group_by = st.radio("Group by", ["category", "channel"], horizontal=True)

    with st.spinner("Computing elasticity…"):
        elas_df = get_elasticity(sales_df_raw, group_by)

    col1, col2 = st.columns(2)
    with col1:
        elas_clean = elas_df.dropna(subset=["elasticity"])
        fig_el = px.bar(
            elas_clean, x=group_by, y="elasticity",
            color="elasticity",
            color_continuous_scale="RdYlGn_r",
            title=f"Price Elasticity by {group_by.title()}",
            labels={"elasticity": "Elasticity Coefficient"},
            text=elas_clean["elasticity"].apply(lambda x: f"{x:.3f}"),
        )
        fig_el.add_hline(y=-1, line_dash="dash", line_color="red",
                          annotation_text="Elastic threshold (−1)")
        fig_el.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_el, use_container_width=True)

    with col2:
        # Avg price vs avg units bubble
        fig_bub = px.scatter(
            elas_clean, x="avg_price", y="avg_units",
            size="n_obs", color=group_by,
            title="Avg Price vs Avg Units Sold",
            color_discrete_sequence=PALETTE,
            labels={"avg_price": "Avg Price ($)", "avg_units": "Avg Units Sold"},
            hover_data=["elasticity", "n_obs"],
        )
        fig_bub.update_layout(height=380)
        st.plotly_chart(fig_bub, use_container_width=True)

    # Table
    st.subheader("📋 Elasticity Summary Table")
    display_cols = [group_by, "elasticity", "ci_lower", "ci_upper",
                    "avg_price", "avg_units", "n_obs", "method"]
    display_cols = [c for c in display_cols if c in elas_df.columns]
    st.dataframe(
        elas_df[display_cols].style.format(
            {c: "{:.3f}" for c in ["elasticity", "ci_lower", "ci_upper",
                                    "avg_price", "avg_units"]
             if c in elas_df.columns}
        ),
        use_container_width=True,
    )

    # Simulation
    st.subheader("🎯 Revenue Impact Simulator")
    sim_cols = st.columns([2, 2, 1])
    with sim_cols[0]:
        sim_cat = st.selectbox(
            "Category to simulate",
            elas_clean["category"].tolist() if "category" in elas_clean.columns
            else elas_clean[group_by].tolist()
        )
    with sim_cols[1]:
        price_change_pct = st.slider("Price change (%)", -30, 30, 10)
    with sim_cols[2]:
        run_sim = st.button("Simulate", type="primary")

    if run_sim:
        from models.price_elasticity import simulate_price_change
        row = elas_clean[elas_clean["category"] == sim_cat].iloc[0]
        sim = simulate_price_change(
            sales_df_raw, sim_cat, row["elasticity"], price_change_pct / 100
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Base Price",    f"${sim['base_price']}")
        s2.metric("New Price",     f"${sim['new_price']}")
        s3.metric("Revenue Change",
                  f"{sim['revenue_change_pct']:+.1f}%",
                  delta_color="normal" if sim["revenue_change_pct"] > 0 else "inverse")
        s4.metric("New Revenue",   f"${sim['new_revenue']:,.0f}")

        # Waterfall chart
        fig_wf = go.Figure(go.Waterfall(
            name="Revenue Impact",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Base Revenue", "Price Effect", "Volume Effect", "New Revenue"],
            y=[
                sim["base_revenue"],
                sim["new_price"] * sim["base_units"] - sim["base_revenue"],
                (sim["new_units"] - sim["base_units"]) * sim["new_price"],
                0,
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#54a24b"}},
            decreasing={"marker": {"color": "#e45756"}},
            totals={"marker": {"color": "#4c78a8"}},
        ))
        fig_wf.update_layout(title=f"Revenue Waterfall — {sim_cat} ({price_change_pct:+d}% price)",
                              height=350)
        st.plotly_chart(fig_wf, use_container_width=True)
