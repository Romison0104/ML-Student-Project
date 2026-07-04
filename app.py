import streamlit as st
import pickle
import gzip
import pandas as pd

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

# ============================================================
# Load Model & Feature Names (cached)
# ============================================================
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    with open("feature_names.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    feature_names = load_feature_names()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("This usually means the scikit-learn version in requirements.txt "
            "does not match the version used to train the model in Colab.")
    st.stop()

# ============================================================
# Header
# ============================================================
st.title("🏦 Bank Term Deposit Predictor")
st.markdown("**COM763 Advanced Machine Learning | Romison Ravichandran**")
st.markdown("---")
st.markdown(
    "Enter customer details below to predict whether they will "
    "subscribe to a term deposit."
)

st.subheader("📋 Customer Information")

# ============================================================
# Input Form
# ============================================================
col1, col2 = st.columns(2)

with col1:
    age                   = st.slider("Age", 18, 85, 40)
    annual_income         = st.number_input("Annual Income (£)", 10000, 300000, 50000, step=1000)
    credit_score          = st.slider("Credit Score", 300, 900, 620)
    account_balance       = st.number_input("Account Balance (£)", 0, 200000, 15000, step=500)
    num_bank_products     = st.slider("Number of Bank Products", 1, 8, 3)
    last_contact_duration = st.slider("Last Contact Duration (seconds)", 0, 600, 200)

with col2:
    net_worth             = st.number_input("Net Worth (£)", 5000, 1600000, 60000, step=1000)
    credit_limit          = st.number_input("Credit Limit (£)", 10000, 120000, 25000, step=500)
    investment_portfolio  = st.number_input("Investment Portfolio (£)", 0, 500000, 30000, step=1000)
    marketing_score       = st.slider("Marketing Score", 0.0, 30.0, 14.0, step=0.1)
    response_propensity   = st.slider("Response Propensity", 0.0, 1.0, 0.3, step=0.01)
    call_response_score   = st.slider("Call Response Score", 0.0, 100.0, 30.0, step=0.5)

st.markdown("---")

# ============================================================
# Prediction
# ============================================================
if st.button("🔮 Predict Subscription", use_container_width=True):

    # Build input row matching training features
    input_dict = {col: 0.0 for col in feature_names}

    raw_values = {
        "Age":                      float(age),
        "AnnualIncome":             float(annual_income),
        "NetWorth":                 float(net_worth),
        "CreditScore":              float(credit_score),
        "CreditLimit":              float(credit_limit),
        "AccountBalance":           float(account_balance),
        "NumBankProducts":          float(num_bank_products),
        "InvestmentPortfolioValue": float(investment_portfolio),
        "MarketingScore":           float(marketing_score),
        "ResponsePropensity":       float(response_propensity),
        "LastContactDuration":      float(last_contact_duration),
        "CallResponseScore":        float(call_response_score),
    }

    for col, val in raw_values.items():
        if col in input_dict:
            input_dict[col] = val

    input_df = pd.DataFrame([input_dict], columns=feature_names).astype("float64")

    # ------------------------------------------------------------
    # Manual z-score scaling using training statistics
    # ------------------------------------------------------------
    numerical_means = {
        "Age": 40.66, "AnnualIncome": 48468.18, "NetWorth": 62767.75,
        "CreditScore": 620.41, "CreditLimit": 26939.26,
        "AccountLengthYears": 6.72, "TenureWithBank": 8.86,
        "AccountBalance": 19484.34, "NumBankProducts": 3.20,
        "InvestmentPortfolioValue": 30000.0, "TotalTransactions": 54.0,
        "AvgTransactionValue": 42.0, "NumOnlineTransactions": 50.0,
        "NumMobileAppLogins": 121.0, "BranchVisitFrequency": 1.0,
        "WebsiteActivityScore": 13.90, "LastContactDuration": 200.0,
        "NumContactsInCampaign": 2.5, "NumPrevCampaignContacts": 0.5,
        "CallResponseScore": 30.0, "DaysSinceLastContact": 100.0,
        "PreviousYearDeposit": 0.0, "MarketingScore": 14.0,
        "ResponsePropensity": 0.3
    }

    numerical_stds = {
        "Age": 11.69, "AnnualIncome": 30612.35, "NetWorth": 73961.54,
        "CreditScore": 61.73, "CreditLimit": 10507.54,
        "AccountLengthYears": 4.61, "TenureWithBank": 4.38,
        "AccountBalance": 15681.71, "NumBankProducts": 1.26,
        "InvestmentPortfolioValue": 50000.0, "TotalTransactions": 20.0,
        "AvgTransactionValue": 30.0, "NumOnlineTransactions": 20.0,
        "NumMobileAppLogins": 30.0, "BranchVisitFrequency": 0.8,
        "WebsiteActivityScore": 3.32, "LastContactDuration": 100.0,
        "NumContactsInCampaign": 1.5, "NumPrevCampaignContacts": 0.8,
        "CallResponseScore": 15.0, "DaysSinceLastContact": 150.0,
        "PreviousYearDeposit": 1.0, "MarketingScore": 5.0,
        "ResponsePropensity": 0.15
    }

    for col in numerical_means:
        if col in input_df.columns:
            mean = numerical_means[col]
            std  = numerical_stds[col]
            if std > 0:
                input_df[col] = (input_df[col] - mean) / std

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    prediction  = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ **WILL Subscribe** to Term Deposit")
        st.metric("Subscription Probability", f"{probability*100:.1f}%")
        st.info("💡 This customer shows a high likelihood of subscribing. "
                "Consider prioritising them in the campaign.")
    else:
        st.error("❌ **Will NOT Subscribe** to Term Deposit")
        st.metric("Subscription Probability", f"{probability*100:.1f}%")
        st.info("💡 This customer is unlikely to subscribe. "
                "Consider targeting with different offers first.")

    # ------------------------------------------------------------
    # Arrow-safe Summary (all values cast to string)
    # ------------------------------------------------------------
    with st.expander("📋 View Input Summary"):
        summary = pd.DataFrame({
            "Feature": [
                "Age", "Annual Income", "Net Worth", "Credit Score",
                "Credit Limit", "Account Balance", "Bank Products",
                "Investment Portfolio", "Marketing Score",
                "Response Propensity", "Contact Duration (s)", "Call Response"
            ],
            "Value": [
                str(age), str(annual_income), str(net_worth), str(credit_score),
                str(credit_limit), str(account_balance), str(num_bank_products),
                str(investment_portfolio), str(marketing_score),
                str(response_propensity), str(last_contact_duration),
                str(call_response_score)
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "COM763 Advanced Machine Learning | Wrexham University | "
    "Model: Random Forest | Dataset: Bank Marketing (100,000 records)"
)
