# ieee-fraud-detection-eda
Exploratory data analysis of 590k e-commerce transactions to identify fraud patterns
markdown
# IEEE-CIS Fraud Detection — Exploratory Data Analysis

**Tools:** Python · pandas · matplotlib · seaborn  
**Dataset:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data)  
**Goal:** Identify transaction patterns that distinguish fraudulent from legitimate payments — supporting data-driven risk decisions.

---

## Key Questions Explored

1. What is the overall fraud rate, and how imbalanced is the dataset?
2. Which transaction amount ranges carry the highest fraud risk?
3. Do certain product categories attract more fraud?
4. Are there time-of-day patterns in fraudulent activity?
5. Which email domains are associated with higher fraud rates?

---

## Key Findings

| Finding | Insight |
|---|---|
| Fraud rate | 3.50% of 590,540 transactions — heavily imbalanced dataset |
| High-risk amounts | Small transactions ($0–25) show the highest fraud rate — consistent with card-testing behaviour |
| Product risk | Category C shows significantly higher fraud rate than average |
| Median fraud amount | $75.00 vs $68.50 for legitimate — fraud skews slightly higher |
| Email domains | Certain less common domains are associated with 2–3× the average fraud rate |

---

## Charts Produced

| Chart | Description |
|---|---|
| `1_fraud_overview.png` | Volume of fraud vs legitimate transactions |
| `2_fraud_by_amount.png` | Fraud rate across transaction amount bands |
| `3_fraud_by_product.png` | Fraud rate by product category |
| `4_amount_distribution.png` | Log-scale amount distribution: fraud vs legitimate |
| `5_fraud_by_hour.png` | Fraud rate by hour of day |
| `6_fraud_by_email.png` | Top email domains by fraud rate |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
python fraud_eda.py
```

Place `train_transaction.csv` and `train_identity.csv` (from Kaggle) in the same directory.

---

## About

This project was completed as part of self-directed learning in financial data analytics and fraud pattern analysis.  
**Author:** Kalaivani Mathivanan · [LinkedIn](https://www.linkedin.com/in/kalaivani-mathivanan)
