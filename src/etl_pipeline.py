# src/etl_pipeline.py
import sqlite3
import numpy as np
import pandas as pd
import os

def run_etl():
    print("--- [1/2] Executing ETL Pipeline (Non-Linear Credit Risk Simulation) ---")
    np.random.seed(42)
    n_records = 15000

    loan_ids = np.arange(100001, 100001 + n_records)
    loan_amnt = np.random.choice([5000, 10000, 15000, 20000, 25000, 30000, 35000], size=n_records)
    term = np.random.choice([' 36 months', ' 60 months'], size=n_records, p=[0.7, 0.3])
    int_rate = np.random.uniform(5.32, 24.99, size=n_records)
    grade = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], size=n_records, p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.05])
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], size=n_records, p=[0.4, 0.1, 0.5])
    annual_inc = np.random.lognormal(mean=11.0, sigma=0.5, size=n_records)
    addr_state = np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'OH', 'PA', 'GA'], size=n_records)
    dti = np.random.uniform(2.0, 35.0, size=n_records)
    fico_score = np.random.normal(700, 45, size=n_records).clip(600, 850)
    revol_util = np.random.uniform(0.0, 95.0, size=n_records)
    purpose = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'], size=n_records)
    
    # NON-LINEAR TARGET GENERATION (Compounding Risk factors for Trees)
    # Risk spikes exponentially when FICO < 660 AND DTI > 25
    fico_penalty = np.where(fico_score < 660, 1.2, 0.0)
    dti_penalty = np.where(dti > 25, 0.9, 0.0)
    compounding_risk = fico_penalty * dti_penalty  

    logit = (
        -0.000008 * (annual_inc - 55000) 
        - 0.015 * (fico_score - 700) 
        + 0.03 * dti 
        + 0.05 * int_rate 
        + (term == ' 60 months') * 0.3 
        + compounding_risk
        - 2.2
    )
    
    default_flag = np.random.binomial(1, 1 / (1 + np.exp(-logit)))

    df_loans = pd.DataFrame({
        'loan_id': loan_ids,
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': np.round(int_rate, 2),
        'grade': grade,
        'home_ownership': home_ownership,
        'annual_inc': np.round(annual_inc, 2),
        'addr_state': addr_state,
        'dti': np.round(dti, 2),
        'fico_score': fico_score.astype(int),
        'revol_util': np.round(revol_util, 2),
        'purpose': purpose,
        'default_flag': default_flag
    })

    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/loan_portfolio.db")
    df_loans.to_sql("loans", conn, if_exists="replace", index=False)
    df_loans.to_csv("data/lending_club_cleaned_portfolio.csv", index=False)
    print("✔ Non-linear dataset generated and saved successfully!")
    conn.close()

if __name__ == "__main__":
    run_etl()