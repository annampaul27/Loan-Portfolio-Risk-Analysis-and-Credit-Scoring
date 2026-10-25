# app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Scoring API", version="1.0")

class LoanApplication(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    grade: str
    home_ownership: str
    annual_inc: float
    dti: float
    fico_score: int
    revol_util: float
    purpose: str

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is running successfully."}

@app.post("/predict")
def predict_risk(app_data: LoanApplication):
    logit = -0.000005 * (app_data.annual_inc - 60000) - 0.012 * (app_data.fico_score - 700) + 0.025 * app_data.dti + 0.04 * app_data.int_rate + (0.2 if app_data.term == ' 60 months' else 0) - 1.8
    prob_default = float(1 / (1 + 2.71828 ** (-logit)))
    
    risk_band = "Band 1 (Very Low)" if prob_default < 0.23 else "Band 2 (Low)" if prob_default < 0.30 else "Band 3 (Medium)" if prob_default < 0.37 else "Band 4 (High)" if prob_default < 0.45 else "Band 5 (Very High)"
    decision = "Auto Approve" if prob_default < 0.30 else "Manual Review" if prob_default < 0.42 else "Reject"
    
    return {
        "probability_of_default": round(prob_default, 4),
        "risk_band": risk_band,
        "decision": decision
    }
