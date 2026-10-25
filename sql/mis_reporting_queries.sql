-- ==============================================================================
-- LOAN PORTFOLIO RISK ANALYSIS & MIS REPORTING (SQL SCRIPT)
-- ==============================================================================

-- 1. Overall Default Rate
SELECT ROUND(AVG(default_flag)*100, 2) as overall_default_rate_pct FROM loans;

-- 2. Total Portfolio Exposure
SELECT SUM(loan_amnt) as portfolio_exposure FROM loans;

-- 3. Default Risk by State View
CREATE VIEW IF NOT EXISTS vw_state_risk AS
SELECT addr_state, COUNT(*) as total_loans, SUM(default_flag) as defaults,
       ROUND(AVG(default_flag)*100, 2) as default_rate_pct, ROUND(SUM(loan_amnt), 2) as total_exposure
FROM loans GROUP BY addr_state ORDER BY default_rate_pct DESC;

-- 4. Default Risk by Grade View
CREATE VIEW IF NOT EXISTS vw_grade_risk AS
SELECT grade, COUNT(*) as total_loans, ROUND(AVG(int_rate), 2) as avg_int_rate,
       ROUND(AVG(default_flag)*100, 2) as default_rate_pct
FROM loans GROUP BY grade ORDER BY grade ASC;

-- Additional MIS Queries
SELECT purpose, COUNT(*) as total_loans, ROUND(AVG(default_flag)*100, 2) as default_rate_pct FROM loans GROUP BY purpose;
SELECT home_ownership, ROUND(AVG(default_flag)*100, 2) as default_rate_pct FROM loans GROUP BY home_ownership;
SELECT COUNT(*) as high_risk_borrowers FROM loans WHERE fico_score < 650;
SELECT term, SUM(loan_amnt) as total_exposure FROM loans GROUP BY term;
SELECT default_flag, ROUND(AVG(dti), 2) as avg_dti FROM loans GROUP BY default_flag;
SELECT addr_state, default_rate_pct FROM vw_state_risk LIMIT 3;
