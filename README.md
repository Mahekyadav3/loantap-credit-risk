# 🏦 Loan Default Predictor — LoanTap Credit Risk Analysis

A machine learning project that predicts whether a borrower will **default on a loan** using Logistic Regression. Built to help LoanTap automate credit risk assessment, reduce Non-Performing Assets (NPAs), and optimize lending decisions.

---

## 📌 Problem Statement

LoanTap wants to determine:
- Whether a loan should be **approved or rejected**
- Which applicants are **likely to default**
- How to **reduce bad loans (NPA risk)**
- How repayment terms can be **optimized**

This is a **binary classification problem:**

| Loan Status | Encoding | Meaning |
|---|---|---|
| Fully Paid | 0 | Non-defaulter |
| Charged Off | 1 | Defaulter |

---

## 📂 Dataset

- **Source:** LoanTap internal dataset (~396K records)
- **Records:** ~396,000 loan applications
- **Target Variable:** `loan_status` (binary: 0 or 1)
- **Class Distribution:** ~80% Fully Paid / ~20% Charged Off (imbalanced)

### Key Features

| Feature | Description |
|---|---|
| `loan_amnt` | Requested loan amount |
| `int_rate` | Interest rate assigned to the loan |
| `grade` | LoanTap's internal credit grade (A–G) |
| `term` | Loan duration (36 or 60 months) |
| `emp_length` | Applicant's employment length (years) |
| `annual_inc` | Annual income of the borrower |
| `dti` | Debt-to-income ratio |
| `home_ownership` | Applicant's housing status |
| `revol_util` | Revolving credit utilization rate |
| `pub_rec` | Number of derogatory public records |
| `mort_acc` | Number of mortgage accounts |
| `earliest_cr_line` | Date of earliest credit line |

---

## 🔍 Project Workflow

```
Data Loading & Exploration
        ↓
Data Cleaning & Feature Engineering
  (term, emp_length conversion, date extraction,
   binary flags, median imputation, column drops)
        ↓
Exploratory Data Analysis (Univariate + Bivariate)
        ↓
Correlation & Multicollinearity Check (VIF)
        ↓
Encoding (Target + One-Hot for Categoricals)
        ↓
Train-Test Split (80/20, Stratified) + StandardScaler
        ↓
Logistic Regression Model Training
        ↓
Model Evaluation
  (Classification Report, Confusion Matrix, ROC-AUC)
        ↓
Precision-Recall Curve & Threshold Tuning
        ↓
Feature Importance Analysis
        ↓
Business Recommendations
```

---

## 🛠️ Feature Engineering

| Transformation | Details |
|---|---|
| `term` | Stripped ' months', converted to int |
| `emp_length` | Mapped `< 1 year` → 0, `10+ years` → 10, extracted digits |
| `issue_d` / `earliest_cr_line` | Parsed to datetime, extracted month & year |
| `credit_history_length` | `issue_year` − `earliest_cr_line_year` |
| `pub_rec_flag` | 1 if `pub_rec > 0` else 0 |
| `mort_acc_flag` | 1 if `mort_acc > 0` else 0 |
| `bankruptcy_flag` | 1 if `pub_rec_bankruptcies > 0` else 0 |
| Missing values | Median imputation on all numerical columns |
| Dropped columns | `emp_title`, `title`, `address`, `sub_grade`, `installment`, `bankruptcy_flag` |

---

## 📊 EDA Highlights

- **Loan amount** distribution is right-skewed — most applications are in the ₹5K–₹20K range
- **Grade A/B** borrowers have significantly lower default rates than Grade E/F/G
- **60-month loans** carry higher default risk than 36-month loans
- **Higher annual income** correlates with lower default probability
- **High revolving utilization** signals financial stress and increased default risk
- Strong positive correlation between `loan_amnt` and `installment` → `installment` dropped (VIF > 97)

---

## 🤖 Modeling

### Logistic Regression
- Solver: `liblinear` (suitable for binary classification)
- Features scaled with `StandardScaler`
- Stratified train-test split to preserve class ratio

### Handling Class Imbalance
- Stratified split ensures 80/20 class ratio is maintained in both sets
- Threshold tuning explored (default 0.5 → custom 0.35) to improve recall on minority class
- Precision-Recall curve used alongside ROC to evaluate imbalanced performance

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| ROC AUC Score | ~0.72 |
| Precision (Class 1) | Lower — model tends to miss defaulters |
| Recall (Class 1) | Low at default threshold; improves with threshold tuning |
| True Negatives | 62,632 (correctly predicted Fully Paid) |
| True Positives | 1,356 (correctly predicted Charged Off) |
| False Negatives | 14,179 (missed defaulters — key risk area) |

> ⚠️ **Class imbalance** is the primary challenge. The high number of False Negatives (missed defaulters) signals that further work with resampling or ensemble methods is needed.

---

## 💡 Key Findings — Feature Importance

**Features increasing default risk (positive coefficients):**
- Higher `int_rate`
- Longer `term` (60 months)
- Lower loan `grade` (E, F, G)
- Higher `revol_util`

**Features reducing default risk (negative coefficients):**
- Higher `annual_inc`
- Better loan `grade` (A, B)
- Longer `credit_history_length`

---

## ✅ Business Recommendations

| # | Recommendation | Business Impact |
|---|---|---|
| 1 | Fast-track approvals for Grade A/B applicants | Grow low-risk loan portfolio |
| 2 | Apply stricter scrutiny to 60-month loans | Reduce long-term default exposure |
| 3 | Use DTI as a primary screening variable | Early detection of high-risk applicants |
| 4 | Implement risk-based interest rate pricing | Align pricing with borrower risk profile |
| 5 | Monitor borrowers with high revolving utilization | Early intervention to prevent defaults |
| 6 | Increase verification for low-income, high-DTI profiles | Reduce fraudulent or risky approvals |

---

## 🔮 Future Enhancements

- **SMOTE / ADASYN** — Oversample minority class to improve recall on defaulters
- **Ensemble Models** — Try Random Forest, XGBoost for better imbalance handling
- **Cost-Sensitive Learning** — Assign higher penalty for missing defaulters
- **Feature Engineering** — Add external credit bureau data, loan purpose clusters
- **Production Deployment** — Wrap model in a Flask/Streamlit app for real-time scoring

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Logistic Regression, scaling, metrics, imputation |
| `statsmodels` | VIF multicollinearity check |
| `scipy` | Statistical analysis |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy jupyter
```

### Run the Notebook
```bash
git clone https://github.com/Mahekyadav3/loantap-credit-risk.git
cd loantap-credit-risk
jupyter notebook Logistic_Reg_LoanTap.ipynb
```

---

## 📁 Repository Structure

```
loantap-credit-risk/
│
├── Logistic_Reg_LoanTap.ipynb     # Main analysis notebook
├── README.md                      # Project documentation
├── FINDINGS.md                    # Detailed insights & recommendations
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git hygiene
```

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 🙋 Author

**Mahek Yadav**

[![GitHub](https://img.shields.io/badge/GitHub-Mahekyadav3-181717?style=flat&logo=github)](https://github.com/Mahekyadav3)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mahek--yadav-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/mahek-yadav-b48b13358)
[![Email](https://img.shields.io/badge/Email-yadavmahek00@gmail.com-D14836?style=flat&logo=gmail)](mailto:yadavmahek00@gmail.com)
