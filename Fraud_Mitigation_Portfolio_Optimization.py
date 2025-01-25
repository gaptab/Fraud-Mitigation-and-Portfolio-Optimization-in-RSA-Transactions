import pandas as pd
import numpy as np

# Creating dummy portfolio data
np.random.seed(42)
portfolio_data = pd.DataFrame({
    "TransactionID": range(1, 1001),
    "CustomerID": np.random.randint(1000, 2000, size=1000),
    "TransactionAmount": np.random.uniform(10, 10000, size=1000),
    "Country": np.random.choice(["UK", "Germany", "France", "Spain"], size=1000),
    "Fraudulent": np.random.choice([0, 1], size=1000, p=[0.95, 0.05]),
    "TransactionDate": pd.date_range(start="2023-01-01", periods=1000, freq="H")
})

# Display first few rows
print(portfolio_data.head())

# Define fraud detection rules
def apply_fraud_rules(df):
    df["Rule_HighValue"] = df["TransactionAmount"] > 5000
    df["Rule_SuspiciousCountry"] = df["Country"].isin(["Germany", "Spain"])
    df["Rule_FrequentTransactions"] = df.groupby("CustomerID")["TransactionID"].transform('count') > 5
    df["FraudDetected"] = (
        df["Rule_HighValue"] | 
        df["Rule_SuspiciousCountry"] | 
        df["Rule_FrequentTransactions"]
    ).astype(int)
    return df

# Apply rules
portfolio_data = apply_fraud_rules(portfolio_data)

# Calculate fraud reduction
fraud_reduction = portfolio_data["Fraudulent"].sum() - portfolio_data["FraudDetected"].sum()
print(f"Fraud reduced by applying new rules: {fraud_reduction} cases")

# Generate KPI report
kpi_report = portfolio_data.groupby("Country").agg(
    TotalTransactions=("TransactionID", "count"),
    FraudulentTransactions=("Fraudulent", "sum"),
    DetectedFraud=("FraudDetected", "sum"),
    AverageTransactionAmount=("TransactionAmount", "mean")
).reset_index()

# Save KPI report
kpi_report.to_csv("RSA_KPI_Report.csv", index=False)

# Display report
print(kpi_report)

import matplotlib.pyplot as plt

# Monthly fraud trends
portfolio_data["Month"] = portfolio_data["TransactionDate"].dt.to_period("M")
monthly_fraud = portfolio_data.groupby("Month").agg(
    Fraudulent=("Fraudulent", "sum"),
    DetectedFraud=("FraudDetected", "sum")
).reset_index()

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(monthly_fraud["Month"].astype(str), monthly_fraud["Fraudulent"], label="Actual Fraud")
plt.plot(monthly_fraud["Month"].astype(str), monthly_fraud["DetectedFraud"], label="Detected Fraud")
plt.title("Fraud Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Cases")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Fraud_Trends.png")
plt.show()
