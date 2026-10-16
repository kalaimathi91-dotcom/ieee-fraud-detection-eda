"""
IEEE-CIS Fraud Detection — Exploratory Data Analysis
Author: Kalaivani Mathivanan
Dataset: https://www.kaggle.com/competitions/ieee-fraud-detection/data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE    = "train_identity.csv"
OUT_DIR          = "outputs/"
import os; os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {"fraud": "#C0392B", "legit": "#2980B9", "neutral": "#7F8C8D"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
txn = pd.read_csv(TRANSACTION_FILE)
idn = pd.read_csv(IDENTITY_FILE)
df  = txn.merge(idn, on="TransactionID", how="left")
print(f"  Transactions: {len(df):,}  |  Fraud rate: {df.isFraud.mean():.2%}")

# ── 1. Overall fraud rate ────────────────────────────────────────────────
fraud_counts = df["isFraud"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Legitimate", "Fraud"], fraud_counts.values,
       color=[PALETTE["legit"], PALETTE["fraud"]], width=0.5)
for i, v in enumerate(fraud_counts.values):
    ax.text(i, v + 500, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
ax.set_title("Transaction Volume: Fraud vs Legitimate", fontweight="bold")
ax.set_ylabel("Number of Transactions")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}1_fraud_overview.png", dpi=150)
plt.close()
print("  ✓ Chart 1: Fraud overview")

# ── 2. Fraud rate by transaction amount band ─────────────────────────────
df["AmtBand"] = pd.cut(df["TransactionAmt"],
    bins=[0, 25, 50, 100, 200, 500, 1000, 10000, df["TransactionAmt"].max()],
    labels=["$0–25","$25–50","$50–100","$100–200","$200–500","$500–1k","$1k–10k","$10k+"])
amt_fraud = df.groupby("AmtBand", observed=True)["isFraud"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(amt_fraud["AmtBand"].astype(str), amt_fraud["isFraud"] * 100,
              color=PALETTE["fraud"], alpha=0.85)
ax.set_title("Fraud Rate by Transaction Amount", fontweight="bold")
ax.set_xlabel("Transaction Amount Band")
ax.set_ylabel("Fraud Rate (%)")
for bar, val in zip(bars, amt_fraud["isFraud"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1%}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}2_fraud_by_amount.png", dpi=150)
plt.close()
print("  ✓ Chart 2: Fraud by amount band")

# ── 3. Fraud rate by product category ───────────────────────────────────
prod_fraud = df.groupby("ProductCD")["isFraud"].agg(["mean","count"]).reset_index()
prod_fraud.columns = ["ProductCD","FraudRate","Count"]
prod_fraud = prod_fraud.sort_values("FraudRate", ascending=True)
fig, ax = plt.subplots(figsize=(6, 4))
colors = [PALETTE["fraud"] if r > df["isFraud"].mean() else PALETTE["legit"]
          for r in prod_fraud["FraudRate"]]
ax.barh(prod_fraud["ProductCD"], prod_fraud["FraudRate"] * 100, color=colors)
ax.axvline(df["isFraud"].mean() * 100, color="grey", linestyle="--", linewidth=1, label="Avg fraud rate")
ax.set_title("Fraud Rate by Product Category", fontweight="bold")
ax.set_xlabel("Fraud Rate (%)")
ax.legend(fontsize=9)
for i, (_, row) in enumerate(prod_fraud.iterrows()):
    ax.text(row["FraudRate"]*100 + 0.1, i, f"{row['FraudRate']:.1%}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}3_fraud_by_product.png", dpi=150)
plt.close()
print("  ✓ Chart 3: Fraud by product")

# ── 4. Transaction amount distribution: fraud vs legit ───────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color, mask in [("Legitimate", PALETTE["legit"], df.isFraud==0),
                             ("Fraud",     PALETTE["fraud"], df.isFraud==1)]:
    vals = np.log1p(df.loc[mask, "TransactionAmt"])
    ax.hist(vals, bins=60, alpha=0.55, label=label, color=color, density=True)
ax.set_title("Transaction Amount Distribution (log scale): Fraud vs Legitimate", fontweight="bold")
ax.set_xlabel("log(1 + Transaction Amount)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}4_amount_distribution.png", dpi=150)
plt.close()
print("  ✓ Chart 4: Amount distribution")

# ── 5. Fraud by day of week / time-of-day proxy ──────────────────────────
# TransactionDT is seconds from a reference point — derive day/hour proxy
df["Hour"] = (df["TransactionDT"] // 3600) % 24
df["DayOfWeek"] = (df["TransactionDT"] // 86400) % 7  # 0=day0, proxy

hour_fraud = df.groupby("Hour")["isFraud"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hour_fraud["Hour"], hour_fraud["isFraud"] * 100,
        color=PALETTE["fraud"], linewidth=2, marker="o", markersize=4)
ax.fill_between(hour_fraud["Hour"], hour_fraud["isFraud"] * 100, alpha=0.15, color=PALETTE["fraud"])
ax.axhline(df["isFraud"].mean() * 100, color="grey", linestyle="--", linewidth=1, label="Avg fraud rate")
ax.set_title("Fraud Rate by Hour of Day (proxy)", fontweight="bold")
ax.set_xlabel("Hour")
ax.set_ylabel("Fraud Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}5_fraud_by_hour.png", dpi=150)
plt.close()
print("  ✓ Chart 5: Fraud by hour")

# ── 6. Top email domains by fraud rate (min 50 transactions) ────────────
if "P_emaildomain" in df.columns:
    email_fraud = df.groupby("P_emaildomain")["isFraud"].agg(["mean","count"]).reset_index()
    email_fraud.columns = ["Domain","FraudRate","Count"]
    email_fraud = email_fraud[email_fraud["Count"] >= 50].sort_values("FraudRate", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(email_fraud["Domain"][::-1], email_fraud["FraudRate"][::-1] * 100, color=PALETTE["fraud"], alpha=0.8)
    ax.axvline(df["isFraud"].mean() * 100, color="grey", linestyle="--", linewidth=1, label="Avg fraud rate")
    ax.set_title("Top 15 Email Domains by Fraud Rate (min 50 transactions)", fontweight="bold")
    ax.set_xlabel("Fraud Rate (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}6_fraud_by_email.png", dpi=150)
    plt.close()
    print("  ✓ Chart 6: Fraud by email domain")

# ── 7. Summary stats table ───────────────────────────────────────────────
summary = pd.DataFrame({
    "Metric": [
        "Total transactions",
        "Fraudulent transactions",
        "Legitimate transactions",
        "Overall fraud rate",
        "Median fraud transaction ($)",
        "Median legitimate transaction ($)",
        "Most fraud-prone product",
        "Highest-risk amount band",
    ],
    "Value": [
        f"{len(df):,}",
        f"{df.isFraud.sum():,}",
        f"{(df.isFraud==0).sum():,}",
        f"{df.isFraud.mean():.2%}",
        f"${df.loc[df.isFraud==1,'TransactionAmt'].median():.2f}",
        f"${df.loc[df.isFraud==0,'TransactionAmt'].median():.2f}",
        df.groupby('ProductCD')['isFraud'].mean().idxmax(),
        str(df.groupby('AmtBand', observed=True)['isFraud'].mean().idxmax()),
    ]
})
summary.to_csv(f"{OUT_DIR}summary_stats.csv", index=False)
print("  ✓ Summary stats saved")

print(f"\nAll outputs saved to ./{OUT_DIR}")
print("\n── Key findings ──────────────────────────────────────────────────")
print(summary.to_string(index=False))
