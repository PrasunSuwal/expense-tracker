from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
client = MongoClient(
    "mongodb+srv://dishantghimire10:3hsQ2m4JAJlNKDO8@expensetracker.87kcsh4.mongodb.net/?retryWrites=true&w=majority&appName=expenseTracker"
)
db = client['test']

# ----------------------
# Helper functions
# ----------------------
def fetch_data(user_id: str, months=6):
    """
    Fetch income and expense data from MongoDB and normalize fields.
    Uses UTC to match MongoDB ISODate. Returns combined list of transactions.
    """
    try:
        from bson import ObjectId
        user_obj_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error converting userId to ObjectId: {e}")
        return []

    date_cutoff = datetime.utcnow() - relativedelta(months=months)  # UTC
    query = {
        "userId": user_obj_id,  # Using ObjectId for userId
        "date": {"$gte": date_cutoff}
    }
    print(f"Debug: Query = {query}")
    print(f"Debug: Looking for user {user_id} with data after {date_cutoff}")

    # Income
    income_cursor = db['incomes'].find(query)
    income_count = db['incomes'].count_documents(query)
    print(f"Debug: Found {income_count} income records")
    
    income_data = []
    for item in income_cursor:
        try:
            if "amount" in item and item["amount"] > 0:
                # Convert string date to datetime if needed
                date = item["date"]
                if isinstance(date, str):
                    date = datetime.strptime(date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                
                income_data.append({
                    "date": date,
                    "amount": float(item["amount"]),
                    "type": "revenue",
                    "category": item.get("source", "Other"),  # Use source field for income
                    "userId": str(item["userId"]),  # Convert ObjectId to string
                    "created": item.get("createdAt"),
                    "updated": item.get("updatedAt")
                })
                print(f"Debug: Processing income: date={date}, amount={item['amount']}, source={item.get('source')}")
        except Exception as e:
            print(f"Error processing income record: {e}, Record: {item}")

    # Expense
    expense_cursor = db['expenses'].find(query)
    expense_count = db['expenses'].count_documents(query)
    print(f"Debug: Found {expense_count} expense records")
    
    expense_data = []
    for item in expense_cursor:
        try:
            if "amount" in item and item["amount"] > 0:
                # Convert string date to datetime if needed
                date = item["date"]
                if isinstance(date, str):
                    try:
                        date = datetime.strptime(date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z")
                
                expense_data.append({
                    "date": date,
                    "amount": float(item["amount"]),
                    "type": "expense",
                    "category": item.get("category", "Other"),
                    "userId": str(item["userId"]),  # Convert ObjectId to string
                    "created": item.get("createdAt"),
                    "updated": item.get("updatedAt")
                })
                print(f"Debug: Processing expense: date={date}, amount={item['amount']}, category={item.get('category')}")
        except Exception as e:
            print(f"Error processing expense record: {e}, Record: {item}")

    combined = income_data + expense_data
    print(f"Fetched {len(combined)} total transactions (income: {len(income_data)}, expenses: {len(expense_data)})")
    return combined

def prepare_dataframe(data):
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    print(f"Debug: DataFrame initial columns: {df.columns.tolist()}")
    print(f"Debug: DataFrame initial shape: {df.shape}")
    
    # Convert date fields
    for date_field in ['date', 'created', 'updated']:
        if date_field in df.columns:
            df[date_field] = pd.to_datetime(df[date_field], utc=True)
    
    # Ensure amount is numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Drop rows with invalid amounts
    df = df.dropna(subset=['amount'])
    
    print(f"Debug: Final DataFrame shape: {df.shape}")
    print(f"Debug: Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Debug: Total amount: {df['amount'].sum()}")
    
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
    now = pd.Timestamp.now(tz='UTC')
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
# API Endpoints
# ----------------------

@app.get("/api/debug/data")
async def debug_data(userId: str):
    """Debug endpoint to check data processing."""
    try:
        data = fetch_data(user_id=userId, months=6)
        df = prepare_dataframe(data)
        
        return {
            "raw_data_count": len(data),
            "dataframe_shape": df.shape,
            "date_range": {
                "start": df['date'].min().isoformat() if not df.empty else None,
                "end": df['date'].max().isoformat() if not df.empty else None
            },
            "total_amount": float(df['amount'].sum()) if not df.empty else 0,
            "sample_records": data[:2] if data else [],
            "columns": df.columns.tolist() if not df.empty else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/collections")
async def debug_collections():
    """Debug endpoint to check what's in the collections."""
    try:
        # Get sample users
        sample_users = list(db['users'].find({}, {"_id": 1, "email": 1}).limit(5))
        
        # Get total counts
        user_count = db['users'].count_documents({})
        income_count = db['incomes'].count_documents({})
        expense_count = db['expenses'].count_documents({})
        
        # Get sample income and expense records
        sample_income = list(db['incomes'].find({}).limit(3))
        sample_expenses = list(db['expenses'].find({}).limit(3))
        
        # Convert ObjectIds to strings for JSON serialization
        import json
        from bson import ObjectId
        
        class JSONEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, ObjectId):
                    return str(o)
                return json.JSONEncoder.default(self, o)
        
        return {
            "collections_count": {
                "users": user_count,
                "incomes": income_count,
                "expenses": expense_count
            },
            "sample_users": [{"_id": str(u["_id"]), "email": u.get("email", "N/A")} for u in sample_users],
            "sample_income": sample_income[:2],
            "sample_expenses": sample_expenses[:2]
        }
    except Exception as e:
        return {"error": str(e)}
@app.api_route("/api/insights", methods=["GET", "POST"])
async def api_insights(request: Request, body: Optional[UserEstimates] = None, userId: str = None):
    if not userId:
        return {"error": "userId is required"}
        
    print(f"Debug: Received userId = {userId}")
    user_estimates = body.estimates if body else {}

    try:
        # Fetch and prepare data
        data = fetch_data(user_id=userId, months=6)
        df = prepare_dataframe(data)
        
        if df.empty:
            print("Debug: No historical data found, using estimates")
            return {
                "forecasts": {
                    "labels": [],
                    "income": [user_estimates.get("income", 0)] * 4,
                    "expenses": [user_estimates.get("expenses", 0)] * 4
                },
                "insights": ["No historical data available. Using provided estimates."],
                "warnings": ["Forecasts are based on estimates only."]
            }
        
        # Generate forecasts
        expense_forecast, expense_labels = forecast_weekly_expenses(df)
        cashflow_labels, cashflow_values = forecast_net_cashflow(df)
        insights, warnings = generate_insights(df)
        
        return {
            "forecasts": {
                "labels": expense_labels,
                "expenses": expense_forecast,
                "cashflow": {
                    "labels": cashflow_labels,
                    "values": cashflow_values
                }
            },
            "insights": insights,
            "warnings": warnings
        }
    except Exception as e:
        print(f"Error in api_insights: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/api/forecast/{userId}")
def api_forecast_get(userId: str):
    try:
        # Fetch last 6 months of data
        data = fetch_data(user_id=userId, months=6)
        print(f"Debug: Fetched {len(data)} records for user {userId}")
        
        df = prepare_dataframe(data)

        if df.empty:
            return {
                "forecast_chart_data": {"labels": [], "categories": {}},
                "cashflow_chart_data": {"labels": [], "values": []},
                "insights": ["No historical data found. Please add some transactions or provide estimates to generate forecasts."],
                "warnings": ["Add your income and expense data to get personalized insights."]
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
    except Exception as e:
        print(f"Error in api_forecast_get: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

@app.post("/api/forecast/{userId}")
def api_forecast_post(userId: str, body: UserEstimates):
    try:
        user_estimates = body.estimates
        print(f"Debug: Received estimates for user {userId}: {user_estimates}")
        
        # Fetch last 6 months of data
        data = fetch_data(user_id=userId, months=6)
        print(f"Debug: Fetched {len(data)} records")
        
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
    except Exception as e:
        print(f"Error in api_forecast_post: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
    
