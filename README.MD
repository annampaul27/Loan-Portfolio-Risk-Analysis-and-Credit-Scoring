# 🏦 Loan Portfolio Risk Analysis & Credit Scoring Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end credit risk analytics and automated underwriting solution built using **Python, SQLite/MySQL, Scikit-learn, FastAPI, Streamlit, and Power BI**. This project simulates real-world retail banking portfolio management using LendingClub-style historical distributions.

---

## 🚀 Project Overview & Highlights

* **ETL & Data Validation Pipeline:** Ingested 15,000+ loan records with built-in data quality checks (null values, duplicates, and cross-table reconciliation).
* **MIS Reporting & SQL Layer:** Developed 20+ analytical queries and views evaluating default risk exposure across states, credit grades, and loan purposes.
* **Credit Scoring Models:** Trained and evaluated **Logistic Regression** and **Random Forest** credit scoring models, measuring performance via **AUC-ROC (0.66)** and **Kolmogorov-Smirnov (KS: ~24%)** statistics. Risk-banded borrowers into 5 distinct credit tiers.
* **Production Deployment:** Delivered a RESTful scoring API via **FastAPI** coupled with an interactive underwriting front end built in **Streamlit**.

---

## 📁 Repository Structure

```text
loan-portfolio-risk-analysis/
│
├── data/
│   ├── loan_portfolio.db                 # SQLite relational database
│   └── lending_club_cleaned_portfolio.csv  # Cleaned dataset exported for Power BI
│
├── sql/
│   └── mis_reporting_queries.sql         # 20+ MIS reporting SQL queries & views
│
├── src/
│   ├── etl_pipeline.py                   # Data ingestion, validation & DB loading
│   └── train_model.py                    # ML model training, evaluation & metrics
│
├── app.py                                # FastAPI backend scoring endpoint
├── app_streamlit.py                      # Streamlit interactive underwriting portal
├── requirements.txt                      # Project Python dependencies
└── README.md                             # Project documentation