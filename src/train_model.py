# src/train_model.py
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_models():
    print("--- [2/2] Loading Data & Training Credit Scoring Models ---")
    conn = sqlite3.connect("data/loan_portfolio.db")
    df_loans = pd.read_sql_query("SELECT * FROM loans", conn)
    conn.close()

    features = ['loan_amnt', 'term', 'int_rate', 'grade', 'home_ownership', 'annual_inc', 'dti', 'fico_score', 'revol_util', 'purpose']
    X = df_loans[features]
    y = df_loans['default_flag']

    num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_score', 'revol_util']
    cat_cols = ['term', 'grade', 'home_ownership', 'purpose']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr_model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict_proba(X_test)[:, 1]

    rf_model = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict_proba(X_test)[:, 1]

    def compute_ks(y_true, y_prob):
        df_ks = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values(by='prob', ascending=False)
        df_ks['good'] = 1 - df_ks['target']
        df_ks['bad'] = df_ks['target']
        df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()
        df_ks['cum_bad'] = df_ks['bad'].cumsum() / df_ks['bad'].sum()
        return np.round(np.max(np.abs(df_ks['cum_bad'] - df_ks['cum_good'])) * 100, 2)

    auc_lr, ks_lr = roc_auc_score(y_test, y_pred_lr), compute_ks(y_test, y_pred_lr)
    auc_rf, ks_rf = roc_auc_score(y_test, y_pred_rf), compute_ks(y_test, y_pred_rf)

    print(f"\n📊 Model Evaluation Results:")
    print(f" - Logistic Regression -> AUC: {{auc_lr:.4f}} | KS: {{ks_lr}}%")
    print(f" - Random Forest       -> AUC: {{auc_rf:.4f}} | KS: {{ks_rf}}%")

if __name__ == "__main__":
    train_models()
