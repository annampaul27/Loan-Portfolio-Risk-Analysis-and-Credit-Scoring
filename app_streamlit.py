# app_streamlit.py
import streamlit as st

st.set_page_config(page_title="Credit Risk Portal", layout="centered")

st.title("🏦 LendingClub Credit Scoring & Underwriting Portal")
st.markdown("Evaluate loan applicant risk profiles, predicted probability of default, and decision bands in real-time.")

st.sidebar.header("Applicant Financial Profile")
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 1000, 40000, 15000, step=500)
term = st.sidebar.selectbox("Term", [" 36 months", " 60 months"])
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 12.5)
grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F"])
annual_inc = st.sidebar.number_input("Annual Income ($)", 20000, 300000, 75000, step=1000)
dti = st.sidebar.slider("Debt-to-Income (DTI)", 0.0, 40.0, 15.0)
fico_score = st.sidebar.slider("FICO Score", 600, 850, 710)
revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0)
purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

if st.button("Run Credit Risk Evaluation", type="primary"):
    logit = -0.000005 * (annual_inc - 60000) - 0.012 * (fico_score - 700) + 0.025 * dti + 0.04 * int_rate + (0.2 if term == ' 60 months' else 0) - 1.8
    prob_default = float(1 / (1 + 2.71828 ** (-logit)))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Default", f"{prob_default * 100:.2f}%")
    with col2:
        band = "Band 1 (Very Low)" if prob_default < 0.23 else "Band 2 (Low)" if prob_default < 0.30 else "Band 3 (Medium)" if prob_default < 0.37 else "Band 4 (High)" if prob_default < 0.45 else "Band 5 (Very High)"
        st.metric("Assigned Risk Band", band)
        
    if prob_default < 0.30:
        st.success("Underwriting Decision: **AUTOMATIC APPROVAL**")
    elif prob_default < 0.42:
        st.warning("Underwriting Decision: **MANUAL UNDERWRITING REVIEW**")
    else:
        st.error("Underwriting Decision: **REJECT (High Default Risk)**")
