# 📋 Detailed Findings & Business Insights

## Project: Loan Default Predictor — LoanTap Credit Risk Analysis

**Author: Mahek Yadav** | [GitHub](https://github.com/Mahekyadav3) · [LinkedIn](https://www.linkedin.com/in/mahek-yadav-b48b13358) · [yadavmahek00@gmail.com](mailto:yadavmahek00@gmail.com)

---

## 1. Dataset Overview

| Property | Detail |
|---|---|
| Total Records | ~396,000 |
| Features (raw) | 27 columns |
| Target Variable | `loan_status` (Fully Paid / Charged Off) |
| Class Distribution | ~80% Fully Paid, ~20% Charged Off |
| Missing Values | Present in `emp_title`, `emp_length`, `mort_acc`, `revol_util`, `pub_rec_bankruptcies` |
| Duplicates | None significant |

**Key Challenge:** Class imbalance — accuracy alone is not a reliable metric. Precision, Recall, F1, and ROC-AUC are used instead.

---

## 2. Feature Engineering Summary

### Conversions
| Column | Original Format | Transformed To |
|---|---|---|
| `term` | `"36 months"` (string) | `36` (integer) |
| `emp_length` | `"< 1 year"`, `"10+ years"` (string) | `0`, `10` (float) |
| `issue_d` | Date string | `issue_month`, `issue_year` (int) |
| `earliest_cr_line` | Date string | `earliest_cr_line_month`, `earliest_cr_line_year` (int) |

### New Engineered Features
| Feature | Logic | Rationale |
|---|---|---|
| `credit_history_length` | `issue_year - earliest_cr_line_year` | Longer credit history = lower risk |
| `pub_rec_flag` | 1 if `pub_rec > 0` else 0 | Binary indicator for derogatory records |
| `mort_acc_flag` | 1 if `mort_acc > 0` else 0 | Binary indicator for mortgage accounts |
| `bankruptcy_flag` | 1 if `pub_rec_bankruptcies > 0` else 0 | Binary indicator for bankruptcy history |

### Dropped Columns & Reasons
| Column | Reason for Dropping |
|---|---|
| `emp_title` | High cardinality text field; not useful for logistic regression |
| `title` | Same as above |
| `address` | Requires geographic feature extraction; out of scope |
| `sub_grade` | Highly correlated with `grade` — redundant |
| `installment` | VIF > 97 — directly derived from `loan_amnt` and `term` |
| `bankruptcy_flag` | VIF > 18 after creation — info captured by `pub_rec_bankruptcies` |

### Missing Value Treatment
All numerical columns imputed using **median strategy** — robust to skewed distributions and outliers.

---

## 3. EDA Findings

### Univariate Analysis

| Feature | Key Observation |
|---|---|
| `loan_amnt` | Right-skewed; most loans are in the ₹5K–₹20K range |
| `int_rate` | Slightly right-skewed; most rates are in the lower-to-mid range |
| `home_ownership` | Majority have MORTGAGE, followed by RENT; fewer OWN outright |
| `loan_status` | ~80% Fully Paid, ~20% Charged Off — imbalanced |

### Bivariate Analysis

| Feature Pair | Insight |
|---|---|
| `loan_amnt` vs `installment` | Strong positive linear relationship — expected, directly correlated |
| `grade` vs `loan_status` | Grade A/B → mostly Fully Paid; Grade E/F/G → significantly higher defaults |
| `term` vs `loan_status` | 60-month loans have higher default proportion than 36-month loans |
| `annual_inc` vs `loan_status` | Higher income → lower default probability (Fully Paid has higher median income) |

### Correlation Highlights
- `loan_amnt` & `installment`: Very high positive correlation (→ `installment` dropped)
- `int_rate` & `grade`: Moderate positive correlation (higher risk grade = higher rate)
- `open_acc` & `total_acc`: Strong positive correlation
- `dti`: Weak correlations with most variables but strong business relevance

---

## 4. Multicollinearity Analysis (VIF)

### Before Treatment
| Feature | VIF |
|---|---|
| `loan_amnt` | 107.65 |
| `installment` | 97.59 |
| `term` | 41.35 |
| `int_rate` | 19.80 |
| `bankruptcy_flag` | 18.25 |
| `pub_rec_bankruptcies` | 13.85 |

### Action Taken
- Dropped `installment` (derived from `loan_amnt` × `term`)
- Dropped `bankruptcy_flag` (derived from `pub_rec_bankruptcies`)
- Retained remaining features after VIF stabilized

---

## 5. Model Results

### Logistic Regression — Confusion Matrix

|  | Predicted Fully Paid | Predicted Charged Off |
|---|---|---|
| **Actual Fully Paid** | 62,632 (TN) ✅ | 1,039 (FP) |
| **Actual Charged Off** | 14,179 (FN) ⚠️ | 1,356 (TP) ✅ |

### Key Metrics
| Metric | Value |
|---|---|
| ROC AUC Score | ~0.72 |
| Recall (Charged Off) | Low at default threshold (0.5) |
| Precision (Charged Off) | Low — many false positives |
| Overall Accuracy | High (but misleading due to imbalance) |

### Threshold Tuning (0.35)
Lowering the classification threshold from 0.50 → 0.35:
- **Recall for Charged Off increases** — model catches more actual defaulters
- **Precision for Charged Off decreases** — more good customers are flagged incorrectly
- **Business decision:** Choose threshold based on whether minimizing NPAs or maximizing approvals is the priority

---

## 6. Feature Importance (Logistic Regression Coefficients)

### Top Features Increasing Default Risk (Positive Coefficients)
| Feature | Business Meaning |
|---|---|
| `int_rate` | Higher interest = higher risk borrowers |
| `term` (60 months) | Longer loan tenure = more exposure |
| `grade_G`, `grade_F`, `grade_E` | Lower grade = higher credit risk |
| `revol_util` | High credit utilization signals financial stress |
| `dti` | Higher debt-to-income = more financial burden |

### Top Features Reducing Default Risk (Negative Coefficients)
| Feature | Business Meaning |
|---|---|
| `annual_inc` | Higher income = greater repayment capacity |
| `grade_A`, `grade_B` | Better grade = more creditworthy |
| `credit_history_length` | Longer history = established creditworthiness |
| `mort_acc_flag` | Having a mortgage account shows financial responsibility |

---

## 7. Business Recommendations

### A. Streamline Low-Risk Approvals
Fast-track applications from **Grade A and B borrowers**. These segments have consistently strong repayment behavior and represent the lowest risk in the portfolio.

### B. Mitigate 60-Month Loan Risk
Apply enhanced scrutiny to 60-month loan requests:
- Require stronger collateral or co-signers
- Apply a risk premium on interest rates
- Implement quarterly monitoring checkpoints

### C. Use DTI as a Primary Screening Variable
Debt-to-income ratio is one of the strongest early signals of default risk. Automate rejection or flagging for applications above a defined DTI threshold (e.g., DTI > 35%).

### D. Risk-Based Pricing Model
Develop an automated pricing engine that assigns interest rates based on predicted default probability:
- **Low Risk (Grade A/B, low DTI):** Competitive rates to attract prime borrowers
- **Medium Risk:** Standard rates with moderate monitoring
- **High Risk:** Higher rates or rejection based on risk score

### E. Early Warning System for High Revolving Utilization
Set up automated alerts for existing borrowers whose revolving utilization crosses a threshold (e.g., > 75%). This enables proactive outreach before borrowers become delinquent.

### F. Threshold Strategy for Business Goals
Decide the operating threshold based on the quarter's objective:
- **Minimize NPAs:** Use threshold 0.35 — catch more defaulters, accept some false positives
- **Maximize approvals:** Use threshold 0.50 — fewer false alarms, but miss some defaulters

---

## 8. Limitations of Current Model

| Limitation | Impact |
|---|---|
| Class imbalance not addressed with resampling | Minority class recall remains low |
| Logistic Regression assumes linearity | May miss complex non-linear patterns |
| No external credit bureau data | Limits credit risk context |
| Static model — no periodic retraining | Drift over time in borrower behavior |
| Threshold chosen manually | Should be optimized using F-beta or cost-sensitive analysis |

---

## 9. Future Enhancements

| Enhancement | Expected Benefit |
|---|---|
| SMOTE / ADASYN oversampling | Better recall on defaulters (minority class) |
| Random Forest / XGBoost | Handles non-linearity and imbalance better |
| Cost-sensitive learning | Directly optimizes for NPA minimization |
| Cross-validation | More robust performance estimates |
| Streamlit / Flask deployment | Real-time credit scoring for loan officers |
| Feature store integration | Enrich with credit bureau, bank statement data |
