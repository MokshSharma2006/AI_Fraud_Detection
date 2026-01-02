import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
 

def get_first_numeric_column(df):
    return df.select_dtypes(include=["int64", "float64"]).columns[0]

credit = pd.read_csv("data/creditcard.csv")
ieee = pd.read_csv("data/fraud_train_preproccessed.csv")
paysim = pd.read_csv("data/paysim.csv")
receiver = pd.read_csv("data/receiver_general.csv")

 
credit_df = credit.copy()
credit_df["transaction_id"] = credit_df.index
credit_df["department"] = "Finance"
credit_df["vendor"] = "Vendor_" + credit_df.index.astype(str)
credit_df["amount"] = credit_df["Amount"]
credit_df["time"] = credit_df["Time"]
credit_df["location"] = np.random.randint(1, 250, len(credit_df))

credit_df = credit_df[
    ["transaction_id", "department", "vendor", "amount", "time", "location"]
]

 
amount_col_ieee = get_first_numeric_column(ieee)

ieee_df = ieee.copy()
ieee_df["transaction_id"] = ieee_df.index
ieee_df["department"] = "Digital Payments"
ieee_df["vendor"] = "Vendor_" + ieee_df.index.astype(str)
ieee_df["amount"] = ieee_df[amount_col_ieee]
ieee_df["time"] = np.arange(len(ieee_df))
ieee_df["location"] = np.random.randint(1, 250, len(ieee_df))

ieee_df = ieee_df[
    ["transaction_id", "department", "vendor", "amount", "time", "location"]
]

paysim_df = paysim.copy()
paysim_df["transaction_id"] = paysim_df.index
paysim_df["department"] = paysim_df["type"]
paysim_df["vendor"] = paysim_df["nameDest"]
paysim_df["amount"] = paysim_df["amount"]
paysim_df["time"] = paysim_df["step"]
paysim_df["location"] = np.random.randint(1, 250, len(paysim_df))

paysim_df = paysim_df[
    ["transaction_id", "department", "vendor", "amount", "time", "location"]
]



receiver_df = receiver.copy()

receiver_df["transaction_id"] = receiver_df.index


receiver_df["department"] = receiver_df[
    receiver_df.columns[0]
]

receiver_df["vendor"] = receiver_df[
    receiver_df.columns[1]
]

receiver_df["amount"] = pd.to_numeric(
    receiver_df[get_first_numeric_column(receiver_df)],
    errors="coerce"
)

receiver_df["time"] = np.arange(len(receiver_df))
receiver_df["location"] = np.random.randint(1, 250, len(receiver_df))

receiver_df = receiver_df[
    ["transaction_id", "department", "vendor", "amount", "time", "location"]
]


df = pd.concat(
    [credit_df, ieee_df, paysim_df, receiver_df],
    ignore_index=True
)


df = df.dropna()
df = df[df["amount"] > 0]


df["vendor_txn_count"] = df.groupby("vendor")["vendor"].transform("count")
df["dept_avg_amount"] = df.groupby("department")["amount"].transform("mean")
df["amount_deviation"] = df["amount"] / df["dept_avg_amount"]

features = df[
    ["amount", "vendor_txn_count", "amount_deviation", "time"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

df["anomaly_score"] = model.decision_function(X_scaled)
df["anomaly_flag"] = model.predict(X_scaled)

df["fraud_risk"] = df["anomaly_score"].apply(
    lambda x: "High" if x < -0.20 else "Medium" if x < 0 else "Low"
)

os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

joblib.dump(model, "model/isolation_forest.pkl")
df.to_csv("output/fraud_results.csv", index=False)

print("✅ Fraud Detection Pipeline Completed Successfully")
print("📁 Output saved to: output/fraud_results.csv")
print("📁 Model saved to: model/isolation_forest.pkl")