import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_feature_names():
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return features

model         = load_model()
feature_names = load_feature_names()

st.title("🏦 Bank Term Deposit Predictor")
st.markdown("**COM763 Advanced Machine Learning | Romison Ravichandran**")
st.markdown("---")
st.markdown(
    "Enter customer details below to predict whether they will "
    "subscribe to a term deposit."
)

st.subheader("📋 Customer Information")

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

if st.button("🔮 Predict Subscription", use_container_width=True):

    # Step 1: Build a row of zeros matching exact training feature names
    input_dict = {col: 0.0 for col in feature_names}

    # Step 2: Fill in raw (unscaled) user values using EXACT column names
    raw_values = {
        "Age":                      age,
        "AnnualIncome":             annual_income,
        "NetWorth":                 net_worth,
        "CreditScore":              credit_score,
        "CreditLimit":              credit_limit,
        "AccountBalance":           account_balance,
        "NumBankProducts":          num_bank_products,
        "InvestmentPortfolioValue": investment_portfolio,
        "MarketingScore":           marketing_score,
        "ResponsePropensity":       response_propensity,
        "LastContactDuration":      last_contact_duration,
        "CallResponseScore":        call_response_score,
    }

    for col, val in raw_values.items():
        if col in input_dict:
            input_dict[col] = val

    # Step 3: Convert to DataFrame — single row, all features
    input_df = pd.DataFrame([input_dict], columns=feature_names)

    # Step 4: Manually scale numerical columns using mean/std from training
    # We scale by computing z-score using known training statistics
    # This avoids the scaler feature name mismatch error entirely
    numerical_means = {
        "Age": 40.66, "AnnualIncome": 48468.18, "NetWorth": 62767.75,
        "CreditScore": 620.41, "CreditLimit": 26939.26,
        "AccountLengthYears": 6.72, "TenureWithBank": 8.86,
        "AccountBalance": 19484.34, "NumBankProducts": 3.20,
        "InvestmentPortfolioValue": 0.0, "TotalTransactions": 0.0,
        "AvgTransactionValue": 0.0, "NumOnlineTransactions": 0.0,
        "NumMobileAppLogins": 0.0, "BranchVisitFrequency": 0.0,
        "WebsiteActivityScore": 13.90, "LastContactDuration": 0.0,
        "NumContactsInCampaign": 0.0, "NumPrevCampaignContacts": 0.0,
        "CallResponseScore": 0.0, "DaysSinceLastContact": 0.0,
        "PreviousYearDeposit": 0.0, "MarketingScore": 0.0,
        "ResponsePropensity": 0.0
    }

    numerical_stds = {
        "Age": 11.69, "AnnualIncome": 30612.35, "NetWorth": 73961.54,
        "CreditScore": 61.73, "CreditLimit": 10507.54,
        "AccountLengthYears": 4.61, "TenureWithBank": 4.38,
        "AccountBalance": 15681.71, "NumBankProducts": 1.26,
        "InvestmentPortfolioValue": 1.0, "TotalTransactions": 1.0,
        "AvgTransactionValue": 1.0, "NumOnlineTransactions": 1.0,
        "NumMobileAppLogins": 1.0, "BranchVisitFrequency": 1.0,
        "WebsiteActivityScore": 3.32, "LastContactDuration": 1.0,
        "NumContactsInCampaign": 1.0, "NumPrevCampaignContacts": 1.0,
        "CallResponseScore": 1.0, "DaysSinceLastContact": 1.0,
        "PreviousYearDeposit": 1.0, "MarketingScore": 1.0,
        "ResponsePropensity": 1.0
    }

    for col in numerical_means:
        if col in input_df.columns:
            mean = numerical_means[col]
            std  = numerical_stds[col]
            if std > 0:
                input_df[col] = (input_df[col] - mean) / std

    # Step 5: Predict
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

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

    with st.expander("📋 View Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Age","Annual Income","Net Worth","Credit Score",
                        "Credit Limit","Account Balance","Bank Products",
                        "Investment Portfolio","Marketing Score",
                        "Response Propensity","Contact Duration","Call Response"],
            "Value":   [age, f"£{annual_income:,}", f"£{net_worth:,}",
                        credit_score, f"£{credit_limit:,}", f"£{account_balance:,}",
                        num_bank_products, f"£{investment_portfolio:,}",
                        marketing_score, response_propensity,
                        f"{last_contact_duration}s", call_response_score]
        })
        st.table(summary)

st.markdown("---")
st.caption(
    "COM763 Advanced Machine Learning | Wrexham University | "
    "Model: Random Forest | Dataset: Bank Marketing (100,000 records)"
)
