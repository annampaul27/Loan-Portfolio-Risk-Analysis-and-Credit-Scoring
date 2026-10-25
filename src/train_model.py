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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def train_advanced_models():
    print("--- [Advanced ML Pipeline] Loading Data & Training ---")
    conn = sqlite3.connect("data/loan_portfolio.db")
    df_loans = pd.read_sql_query("SELECT * FROM loans", conn)
    conn.close()

    # Feature Engineering
    df_loans['income_to_loan_ratio'] = df_loans['annual_inc'] / df_loans['loan_amnt']
    df_loans['int_rate_band'] = pd.qcut(df_loans['int_rate'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df_loans['grade_term'] = df_loans['grade'].astype(str) + "_" + df_loans['term'].astype(str)

    features = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'home_ownership', 
        'annual_inc', 'dti', 'fico_score', 'revol_util', 'purpose',
        'income_to_loan_ratio', 'int_rate_band', 'grade_term'
    ]
    
    X = df_loans[features]
    y = df_loans['default_flag']

    num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_score', 'revol_util', 'income_to_loan_ratio']
    cat_cols = ['term', 'grade', 'home_ownership', 'purpose', 'int_rate_band', 'grade_term']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Baseline Logistic Regression
    lr_pipeline = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    # Optimized XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    xgb_pipeline = Pipeline([
        ('prep', preprocessor), 
        ('clf', XGBClassifier(
            n_estimators=200, 
            learning_rate=0.03, 
            max_depth=6,             # Deeper trees to capture compounding risk
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

    def compute_ks(y_true, y_prob):
        df_ks = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values(by='prob', ascending=False)
        df_ks['good'] = 1 - df_ks['target']
        df_ks['bad'] = df_ks['target']
        df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()
        df_ks['cum_bad'] = df_ks['bad'].cumsum() / df_ks['bad'].sum()
        return np.round(np.max(np.abs(df_ks['cum_bad'] - df_ks['cum_good'])) * 100, 2)

    auc_lr, ks_lr = roc_auc_score(y_test, y_pred_lr), compute_ks(y_test, y_pred_lr)
    auc_xgb, ks_xgb = roc_auc_score(y_test, y_pred_xgb), compute_ks(y_test, y_pred_xgb)

    print(f"\n📊 Optimized Model Evaluation Results:")
    print(f" - Logistic Regression -> AUC: {auc_lr:.4f} | KS: {ks_lr}%")
    print(f" - Optimized XGBoost   -> AUC: {auc_xgb:.4f} | KS: {ks_xgb}%")

    print(f"\n📋 Updated Resume Bullet Point Candidate:")
    print(f"- Engineered predictive interaction features and trained an optimized XGBoost credit scoring model achieving an AUC of {auc_xgb:.2f} and KS of {ks_xgb}%.")

if __name__ == "__main__":
    train_advanced_models()