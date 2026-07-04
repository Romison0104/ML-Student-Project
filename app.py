import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load model and supporting files ──────────────────────────
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_feature_names():
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return features

model        = load_model()
scaler       = load_scaler()
feature_names = load_feature_names()

# ── App header ────────────────────────────────────────────────
st.title("Bank Term Deposit Predictor")
st.markdown("**COM763 Advanced Machine Learning | Romison Ravichandran**")
st.markdown("---")
st.markdown(
    "Enter customer details below to predict whether they will "
    "subscribe to a term deposit."
)

# ── Input form ────────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age            = st.slider("Age", 18, 85, 40)
    annual_income  = st.number_input("Annual Income (£)", 10000, 300000, 50000, step=1000)
    credit_score   = st.slider("Credit Score", 300, 900, 620)
    account_balance = st.number_input("Account Balance (£)", 0, 200000, 15000, step=500)
    num_bank_products = st.slider("Number of Bank Products", 1, 8, 3)
    last_contact_duration = st.slider("Last Contact Duration (seconds)", 0, 600, 200)

with col2:
    net_worth      = st.number_input("Net Worth (£)", 5000, 1600000, 60000, step=1000)
    credit_limit   = st.number_input("Credit Limit (£)", 10000, 120000, 25000, step=500)
    investment_portfolio = st.number_input("Investment Portfolio (£)", 0, 500000, 30000, step=1000)
    marketing_score = st.slider("Marketing Score", 0.0, 30.0, 14.0, step=0.1)
    response_propensity = st.slider("Response Propensity", 0.0, 1.0, 0.3, step=0.01)
    call_response_score = st.slider("Call Response Score", 0.0, 100.0, 30.0, step=0.5)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────
if st.button("Predict Subscription", use_container_width=True):

    # Build input row with all zeros first
    input_dict = {col: 0 for col in feature_names}

    # Fill in the values the user provided
    input_dict["Age"]                     = age
    input_dict["AnnualIncome"]            = annual_income
    input_dict["NetWorth"]                = net_worth
    input_dict["CreditScore"]             = credit_score
    input_dict["CreditLimit"]             = credit_limit
    input_dict["AccountBalance"]          = account_balance
    input_dict["NumBankProducts"]         = num_bank_products
    input_dict["InvestmentPortfolioValue"] = investment_portfolio
    input_dict["MarketingScore"]          = marketing_score
    input_dict["ResponsePropensity"]      = response_propensity
    input_dict["LastContactDuration"]     = last_contact_duration
    input_dict["CallResponseScore"]       = call_response_score

    # Convert to dataframe and scale numerical columns
    input_df = pd.DataFrame([input_dict])

    cols_to_scale = ["Age","AnnualIncome","NetWorth","CreditScore",
                     "CreditLimit","AccountLengthYears","TenureWithBank",
                     "AccountBalance","NumBankProducts","InvestmentPortfolioValue",
                     "TotalTransactions","AvgTransactionValue",
                     "NumOnlineTransactions","NumMobileAppLogins",
                     "BranchVisitFrequency","WebsiteActivityScore",
                     "LastContactDuration","NumContactsInCampaign",
                     "NumPrevCampaignContacts","CallResponseScore",
                     "DaysSinceLastContact","PreviousYearDeposit",
                     "MarketingScore","ResponsePropensity"]

    # Only scale columns that exist in input_df
    scale_these = [c for c in cols_to_scale if c in input_df.columns]
    input_df[scale_these] = scaler.transform(input_df[scale_these])

    # Predict
    prediction   = model.predict(input_df)[0]
    probability  = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"**WILL Subscribe** to Term Deposit")
        st.metric("Subscription Probability", f"{probability*100:.1f}%")
        st.info("💡 This customer shows a high likelihood of subscribing. "
                "Consider prioritising them in the campaign.")
    else:
        st.error(f"**Will NOT Subscribe** to Term Deposit")
        st.metric("Subscription Probability", f"{probability*100:.1f}%")
        st.info("This customer is unlikely to subscribe. "
                "Consider targeting with different offers first.")

    # Show input summary
    with st.expander("View Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Age","Annual Income","Net Worth","Credit Score",
                        "Credit Limit","Account Balance","Bank Products",
                        "Investment Portfolio","Marketing Score",
                        "Response Propensity","Contact Duration","Call Response"],
            "Value": [age, f"£{annual_income:,}", f"£{net_worth:,}",
                      credit_score, f"£{credit_limit:,}", f"£{account_balance:,}",
                      num_bank_products, f"£{investment_portfolio:,}",
                      marketing_score, response_propensity,
                      f"{last_contact_duration}s", call_response_score]
        })
        st.table(summary)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "COM763 Advanced Machine Learning | Wrexham University | "
    "Model: Random Forest | Dataset: Bank Marketing (100,000 records)"
)
