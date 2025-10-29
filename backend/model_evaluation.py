import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# MongoDB setup
client = MongoClient("your_mongodb_connection_uri")
db = client["your_database_name"]
expenses_collection = db["expenses"]
incomes_collection = db["incomes"]

# UserId (replace with one in your data)
USER_ID = "6894d9ef0b7d1c9b79430fa0"

# Fetch data (last 6 months)
start_date = datetime(2025, 4, 1)
expenses = list(expenses_collection.find({"userId": USER_ID, "date": {"$gte": start_date}}))
incomes = list(incomes_collection.find({"userId": USER_ID, "date": {"$gte": start_date}}))

# Convert to DataFrame
expenses_df = pd.DataFrame(expenses)
incomes_df = pd.DataFrame(incomes)

# Combine into a single timeline
if not expenses_df.empty:
    expenses_df["date"] = pd.to_datetime(expenses_df["date"])
    daily_expense = expenses_df.groupby("date")["amount"].sum()

if not incomes_df.empty:
    incomes_df["date"] = pd.to_datetime(incomes_df["date"])
    daily_income = incomes_df.groupby("date")["amount"].sum()

combined = pd.DataFrame({
    "income": daily_income,
    "expense": daily_expense
}).fillna(0)

# Target = Net Savings = Income - Expense
combined["net_savings"] = combined["income"] - combined["expense"]

# Sort by date
combined = combined.sort_index()

# Split train-test (80-20)
train_size = int(len(combined) * 0.8)
train = combined.iloc[:train_size]
test = combined.iloc[train_size:]

# Train model on training set
model = ExponentialSmoothing(train["net_savings"], trend='add', seasonal=None)
fit = model.fit()

# Predict on test set
preds = fit.forecast(len(test))
# Calculate metricssss
mae = mean_absolute_error(test["net_savings"], preds)
rmse = math.sqrt(mean_squared_error(test["net_savings"], preds))
r2 = r2_score(test["net_savings"], preds)

# Confidence Level
if r2 >= 0.85:
    confidence = "High"
elif r2 >= 0.6:
    confidence = "Medium"
else:
    confidence = "Low"

# Print report
print("=== Model Evaluation Report ===")
print(f"Data points used: {len(combined)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
print(f"Confidence Level: {confidence}")
print("\nPredicted vs Actual (last 5 records):")
print(pd.DataFrame({"Actual": test["net_savings"].tail(5), "Predicted": preds.tail(5)}))
