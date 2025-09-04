from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict

# ----------------------
# FastAPI app setup
# ----------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# MongoDB connection
# ----------------------
client = MongoClient("mongodb+srv://dishantghimire10:NPUdOQJJGBPWgTHj@cluster0.wplbfhh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['test']
collection = db['transactions']

# ----------------------
# Helper functions
# ----------------------
def fetch_data(months=6):
    date_cutoff = datetime.now() - timedelta(days=30 * months)
    cursor = collection.find({"date": {"$gte": date_cutoff.strftime("%Y-%m-%d")}})
    return list(cursor)

def prepare_dataframe(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = pd.to_numeric(df['amount'])
    return df

def forecast_weekly_expenses(df):
    expenses = df[df['type'] == 'expense']
    if expenses.empty:
        return {}, []

    weekly = expenses.groupby([pd.Grouper(key='date', freq='W-MON'), 'category'])['amount'].sum().reset_index()
    categories = weekly['category'].unique()
    forecast_results = {}
    last_week = weekly['date'].max()
    weeks_labels = [(last_week + pd.Timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 5)]

    for cat in categories:
        cat_data = weekly[weekly['category'] == cat].set_index('date').sort_index()
        if len(cat_data) < 3:
            last_val = cat_data['amount'].iloc[-1]
            forecast_results[cat] = [last_val] * 4
            continue
        model = ExponentialSmoothing(cat_data['amount'], trend='add', seasonal=None, initialization_method="estimated").fit()
        forecast = model.forecast(4)
        forecast_results[cat] = forecast.round(2).tolist()
    return forecast_results, weeks_labels

def forecast_net_cashflow(df):
    if df.empty:
        return [], []
    df['net'] = np.where(df['type'] == 'revenue', df['amount'], -df['amount'])
    weekly_net = df.groupby(pd.Grouper(key='date', freq='W-MON'))['net'].sum().sort_index()
    if len(weekly_net) < 3:
        last_net = weekly_net.iloc[-1] if len(weekly_net) > 0 else 0
        labels = []
        values = [last_net] * 4
    else:
        model = ExponentialSmoothing(weekly_net, trend='add', seasonal=None, initialization_method="estimated").fit()
        forecast = model.forecast(4)
        labels = [(weekly_net.index.max() + pd.Timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 5)]
        values = forecast.round(2).tolist()
    return labels, values

def generate_insights(df):
    insights = []
    warnings = []
    if df.empty:
        return insights, warnings
    now = pd.Timestamp.now()
    this_month = df[df['date'].dt.month == now.month]
    last_3_months = df[(df['date'] >= now - pd.DateOffset(months=3)) & (df['date'] < now)]

    revenue = this_month[this_month['type'] == 'revenue']['amount'].sum()
    expense = this_month[this_month['type'] == 'expense']['amount'].sum()
    savings = revenue - expense
    savings_rate = (savings / revenue) if revenue > 0 else 0

    if savings_rate < 0.3:
        insights.append(f"Your savings rate this month is {int(savings_rate*100)}%. Try to increase it to at least 40%.")

    cat_avg = last_3_months[last_3_months['type']=='expense'].groupby('category')['amount'].mean()
    cat_this = this_month[this_month['type']=='expense'].groupby('category')['amount'].sum()
    for cat in cat_this.index:
        avg = cat_avg.get(cat, 0)
        curr = cat_this[cat]
        if avg == 0:
            continue
        if curr > 1.5 * avg:
            diff = int(curr - avg)
            insights.append(f"Your expenses in {cat} are {int(((curr/avg)-1)*100)}% above your average. Consider cutting down to save Rs {diff} this month.")

    revenue_by_source = this_month[this_month['type']=='revenue'].groupby('category')['amount'].sum()
    total_revenue = revenue_by_source.sum()
    freelance_share = (revenue_by_source.get('Freelance', 0) / total_revenue) if total_revenue > 0 else 0
    if freelance_share > 0.2:
        insights.append(f"Your income relies {int(freelance_share*100)}% on freelance work. Plan for unpredictability.")

    expense_last_month = last_3_months[last_3_months['type']=='expense']['amount'].sum()
    expense_this_month = expense
    if expense_this_month > expense_last_month * 1.2:
        warnings.append("Your expenses increased sharply this month, you might save less next month if this continues.")

    avg_monthly_expense = last_3_months[last_3_months['type']=='expense'].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().mean()
    emergency_fund_needed = avg_monthly_expense * 3
    emergency_fund_saved = savings if savings > 0 else 0
    if emergency_fund_saved < emergency_fund_needed:
        diff = int(emergency_fund_needed - emergency_fund_saved)
        insights.append(f"You should maintain an emergency fund of Rs {int(emergency_fund_needed)} (3 months expenses). You are Rs {diff} short.")

    if savings > 0:
        months = 12 * 5
        rate = 0.12 / 12
        fv = savings * (((1 + rate) ** months - 1) / rate) * (1 + rate)
        insights.append(f"If you invest Rs {int(savings)}/month in a 12% SIP, you will have approximately Rs {int(fv)} in 5 years.")
    return insights, warnings

# ----------------------
# Request Model
# ----------------------
class UserEstimates(BaseModel):
    estimates: Dict[str, float] = Field(default_factory=dict)

# ----------------------
# API Endpoint
# ----------------------
@app.api_route("/api/insights", methods=["GET", "POST"])
async def api_insights(request: Request, body: Optional[UserEstimates] = None):
    user_estimates = body.estimates if body else {}

    data = fetch_data()
    df = prepare_dataframe(data)

    if df.empty:
        weekly_labels = []
        today = datetime.now()
        for i in range(1, 5):
            weekly_labels.append((today + timedelta(weeks=i)).strftime('%Y-%m-%d'))

        forecast_results = {}
        revenue_sum = 0
        expense_sum = 0

        for cat, amount in user_estimates.items():
            if cat.lower() in ["salary", "freelance"]:
                revenue_sum += amount
            else:
                expense_sum += amount
                forecast_results[cat] = [round(amount / 4, 2)] * 4

        weekly_revenue = revenue_sum / 4
        weekly_expense = expense_sum / 4
        cashflow_values = [round(weekly_revenue - weekly_expense, 2)] * 4

        insights = [
            "No historical data found. Using your input estimates for forecasting.",
            f"Your total monthly income estimate is Rs {int(revenue_sum)}.",
            f"Your total monthly expense estimate is Rs {int(expense_sum)}.",
            "Forecasts will improve as you add real transaction data."
        ]

        return {
            "forecast_chart_data": {"labels": weekly_labels, "categories": forecast_results},
            "cashflow_chart_data": {"labels": weekly_labels, "values": cashflow_values},
            "insights": insights,
            "warnings": []
        }

    forecast_data, forecast_labels = forecast_weekly_expenses(df)
    cashflow_labels, cashflow_values = forecast_net_cashflow(df)
    insights, warnings = generate_insights(df)

    return {
        "forecast_chart_data": {"labels": forecast_labels, "categories": forecast_data},
        "cashflow_chart_data": {"labels": cashflow_labels, "values": cashflow_values},
        "insights": insights,
        "warnings": warnings
    }

