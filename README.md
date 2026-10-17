# Loan Portfolio Early Warning System

Early warning model that flags loan accounts showing early signs of stress, using behavioural indicators and Logistic Regression on LendingClub data.

> Designed as an interpretable monitoring baseline, not a default prediction system.

## 1. Problem Statement

Identify loan accounts showing early signs of stress before default or severe delinquency occurs, so risk teams can monitor them and act early.

## 2. Objective

- Flag risky accounts early for monitoring and preventive action
- Prioritise recall (capturing stressed accounts) over accuracy
- Keep the model interpretable for risk teams

## 3. Data

- **Source:** LendingClub loan data (public, consumer and personal lending)
- **How to get the data:** this project uses the public LendingClub loan dataset (`loan.csv`, 145 columns). The file is too large to store in this repo, so download it from the public source and place it in the same folder as the notebook.
- **Sample:** first 300,000 records of the loan file (not a random sample)
- **Features:** about 16 selected for interpretability and to avoid leakage

## 4. Target Definition (Early Stress)

A loan is labelled early stress (1) if ANY of these apply, else 0:

- 90+ days past due in the last 24 months
- 120+ days past due historically
- Recent revolving delinquency (6 months or less)

A 30+ DPD condition was also included in the code, but that column is all zeros in this sample, so it has no effect on the target. About **22.5%** of accounts are labelled early stress.

## 5. Feature Engineering

- **DTI:** the value 999 was treated as missing income and replaced with NaN; a `dti_missing` flag was added
- **Delinquency signals:** sparse features converted to binary indicators; missing values treated as "no delinquency"
- **Encoding:** `term` converted to numeric; `sub_grade` encoded ordinally to keep the risk order

## 6. Data Leakage Handling

The first model scored a perfect ROC-AUC of 1.0, which signalled leakage. Features used to build the target were still in the model inputs. After removing every target-derived feature, ROC-AUC fell to a realistic 0.64.

> Rule: features used to create the target must not be used to predict it.

## 7. Modelling

- **Model:** Logistic Regression (interpretable, suits risk teams)
- **Preprocessing:** median imputation, StandardScaler
- **Class imbalance:** `class_weight='balanced'`

## 8. Results

ROC-AUC: **0.64** on a 75,000-account test set (20% hold-out).

| Threshold | Recall (stressed) | Precision (stressed) | Share of accounts flagged |
| --------- | ----------------- | -------------------- | ------------------------- |
| 0.5       | about 59%         | about 31%            | lower                     |
| 0.4       | about 89%         | about 26%            | about 78%                 |

The base rate is 22.5%, so precision at 0.4 is only slightly above random. Lowering the threshold finds more stressed accounts but flags most of the portfolio, which is why this is a baseline monitoring model and not production-ready.

## 9. Key Insights

- Higher sub-grade risk strongly increases early stress
- Recent credit activity signals risk
- Credit risk is multi-factor, not driven by one variable

## 10. Business Interpretation

The model suits monitoring and prioritisation, not decisions. It is weak at separating stressed from healthy accounts, so it should be combined with other signals before any action.

## 11. Limitations

- The target is rule-based, not an actual default outcome
- The data is consumer lending, not SME lending
- No time-series payment behaviour
- First 300,000 rows are not a random sample
- Moderate predictive power (AUC 0.64)

## 12. Future Improvements

- Rank accounts into risk deciles and measure lift in the top decile
- Add time-based behavioural features
- Try gradient boosting and compare with Logistic Regression
- Tune the threshold on a separate validation set

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
