# # SME Credit Risk – Early Warning System
Early Warning System for SME credit risk using behavioral indicators and logistic regression
> This project focuses on early risk detection using behavioral indicators and is designed for interpretability and real-world risk monitoring.

## 1. Problem Statement
This project builds an **Early Warning System (EWS)** to identify loan accounts showing **early signs of stress** before default or severe delinquency occurs.

## 2. Objective
To proactively flag risky loan accounts so that:
- Risk teams can monitor accounts early  
- Preventive actions can be taken  
- Credit losses can be reduced  
The model prioritizes **recall (capturing risky accounts)** over accuracy.

## 3. Data Overview
- Source: LendingClub dataset (publicly available)  
- Sample size: ~300,000 records  
The dataset includes:
- Loan characteristics  
- Borrower credit profile  
- Repayment behavior  
Only **~16 features** were selected to ensure:
- Interpretability  
- Reduced noise  
- No data leakage  

## 4. Target Definition (Early Stress)
A loan is classified as **early stress (1)** if ANY:
- 30+ days past due (DPD)  
- 90+ DPD in last 24 months  
- 120+ DPD historically  
- Recent revolving delinquency (≤ 6 months)  
Else → 0  
This enables **early detection** without relying on default labels.

## 5. Feature Engineering
### Key Transformations:
**DTI Handling**
- `dti = 999` treated as missing income  
- Created `dti_missing` flag  
- Replaced 999 with NaN  
**Delinquency Signals**
- Converted sparse features into binary indicators  
- Missing values treated as “no delinquency”  
**Categorical Encoding**
- `term` converted to numeric  
- `sub_grade` encoded ordinally to preserve risk hierarchy
  
## 6. Data Leakage Handling
Initial model showed **perfect performance (ROC-AUC = 1.0)**, indicating leakage.
**Cause:**
Features used to define the target were included in the model  
**Fix:**
Removed all target-derived features from inputs  
> Rule: Features used to create the target must not be used for prediction.

## 7. Modeling Approach
- Model: **Logistic Regression**  
- Reason: Interpretability and suitability for risk teams  
**Preprocessing:**
- Missing values handled using median imputation  
- Features scaled using StandardScaler  
- Class imbalance handled using `class_weight='balanced'`  

## 8. Evaluation
- ROC-AUC ≈ **0.64** (realistic, non-leaky)  
- Recall prioritized over precision  
### Threshold Tuning:
- Default threshold: 0.5  
- Final threshold: **0.4**  
**Result:**
- Recall improved from ~59% → ~89%  
- Higher false positives accepted to reduce missed risk  

## 9. Key Insights
- Higher **sub-grade risk** strongly increases early stress  
- Increased **recent credit activity** signals risk  
- Credit risk is **multi-factor**, not driven by a single variable  

## 10. Business Interpretation
This model is designed as a **monitoring tool**, not a default prediction system.
- False positives → manageable (monitoring)  
- False negatives → costly (missed early stress)  
  
## 11. Limitations
- Rule-based target, not actual default outcome  
- No time-series payment behavior  
- Moderate predictive power expected for early signals  
  
## 12. Future Improvements
- Add time-based behavioral features  
- Use transaction-level payment data  
- Build segment-specific models  
  
## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn
- Matplotlib, Seaborn
- Matplotlib, Seaborn  

---
