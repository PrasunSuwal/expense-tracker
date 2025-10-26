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
                    date = datetime.strptime(
                        date.split('.')[0], "%Y-%m-%dT%H:%M:%S")

                # Validate date range (reject dates before 1900 or after 2100)
                if date.year < 1900 or date.year > 2100:
                    print(f"Debug: Skipping invalid income date: {date}")
                    continue

                income_data.append({
                    "date": date,
                    "amount": float(item["amount"]),
                    "type": "revenue",
                    # Use source field for income
                    "category": item.get("source", "Other"),
                    # Convert ObjectId to string
                    "userId": str(item["userId"]),
                    "created": item.get("createdAt"),
                    "updated": item.get("updatedAt")
                })
                print(
                    f"Debug: Processing income: date={date}, amount={item['amount']}, source={item.get('source')}")
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
                        date = datetime.strptime(
                            date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        date = datetime.strptime(
                            date, "%Y-%m-%dT%H:%M:%S.%f%z")

                # Validate date range (reject dates before 1900 or after 2100)
                if date.year < 1900 or date.year > 2100:
                    print(f"Debug: Skipping invalid date: {date}")
                    continue

                expense_data.append({
                    "date": date,
                    "amount": float(item["amount"]),
                    "type": "expense",
                    "category": item.get("category", "Other"),
                    # Convert ObjectId to string
                    "userId": str(item["userId"]),
                    "created": item.get("createdAt"),
                    "updated": item.get("updatedAt")
                })
                print(
                    f"Debug: Processing expense: date={date}, amount={item['amount']}, category={item.get('category')}")
        except Exception as e:
            print(f"Error processing expense record: {e}, Record: {item}")

    combined = income_data + expense_data
    print(
        f"Fetched {len(combined)} total transactions (income: {len(income_data)}, expenses: {len(expense_data)})")
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


def forecast_monthly_expenses(df):
    """
    Forecast monthly expenses for next 3 months - better for realistic patterns
    """
    expenses = df[df['type'] == 'expense']
    if expenses.empty:
        return {}, []

    # Create monthly aggregation for better pattern recognition
    monthly = expenses.groupby(
        [pd.Grouper(key='date', freq='M'), 'category'])['amount'].sum().reset_index()
    
    # Also keep weekly for frequency analysis
    weekly = expenses.groupby(
        [pd.Grouper(key='date', freq='W-MON'), 'category'])['amount'].sum().reset_index()
    
    # Get total time periods for frequency calculation
    total_months = len(monthly['date'].unique())
    total_weeks = len(weekly['date'].unique())
    categories = monthly['category'].unique()
    forecast_results = {}
    
    # Generate next 3 months labels
    last_month = monthly['date'].max() if not monthly.empty else pd.Timestamp.now()
    months_labels = [(last_month + pd.DateOffset(months=i)
                     ).strftime('%Y-%m') for i in range(1, 4)]

    print(f"Debug: Monthly forecasting for {len(categories)} categories over {total_months} months")

    for cat in categories:
        cat_monthly = monthly[monthly['category'] == cat].set_index('date').sort_index()
        cat_weekly = weekly[weekly['category'] == cat].set_index('date').sort_index() if not weekly.empty else pd.DataFrame()
        
        # Calculate spending patterns with monthly focus
        months_with_spending = len(cat_monthly)
        weeks_with_spending = len(cat_weekly) if not cat_weekly.empty else 0
        monthly_frequency = months_with_spending / total_months if total_months > 0 else 0
        weekly_frequency = weeks_with_spending / total_weeks if total_weeks > 0 else 0
        
        # Monthly analysis
        avg_monthly_amount = cat_monthly['amount'].mean() if len(cat_monthly) > 0 else 0
        max_monthly = cat_monthly['amount'].max() if len(cat_monthly) > 0 else 0
        recent_monthly = cat_monthly['amount'].iloc[-2:].mean() if len(cat_monthly) >= 2 else avg_monthly_amount
        
        print(f"Debug: '{cat}' - Monthly freq: {monthly_frequency:.1%} ({months_with_spending}/{total_months}), Avg: ₹{avg_monthly_amount:.0f}")
        
        # Improved categorization for monthly forecasting
        monthly_bills = ['rent', 'utilities', 'insurance', 'subscription', 'mortgage', 'loan', 'emi']
        essential_monthly = ['groceries', 'food'] if monthly_frequency >= 0.7 else []
        regular_monthly = ['shopping', 'entertainment'] if monthly_frequency >= 0.5 else []
        
        is_monthly_bill = cat.lower() in monthly_bills or monthly_frequency >= 0.8
        is_essential_monthly = cat.lower() in essential_monthly or (monthly_frequency >= 0.7 and avg_monthly_amount > 2000)
        is_regular_monthly = cat.lower() in regular_monthly or monthly_frequency >= 0.5
        is_occasional_monthly = 0.2 <= monthly_frequency < 0.5
        is_rare_monthly = monthly_frequency < 0.2
        
        # Detect trends in recent vs older data
        if len(cat_monthly) >= 4:
            recent_trend = cat_monthly['amount'].tail(2).mean()
            older_trend = cat_monthly['amount'].head(2).mean()
            trend_change = (recent_trend - older_trend) / older_trend if older_trend > 0 else 0
        else:
            trend_change = 0
            
        print(f"Debug: '{cat}' trend: {trend_change:.1%}")
        
        # Monthly forecasting logic
        if is_monthly_bill:
            # Monthly bills: Appear every month with trend adjustment
            base_amount = recent_monthly if recent_monthly > 0 else avg_monthly_amount
            
            # Apply conservative trend for bills
            if trend_change > 0.1:
                base_amount *= 1.05  # 5% increase
            elif trend_change < -0.1:
                base_amount *= 0.95  # 5% decrease
                
            forecast_results[cat] = [base_amount] * 3  # Every month
            print(f"Debug: '{cat}' classified as MONTHLY BILL")
            
        elif is_essential_monthly:
            # Essential monthly: Every month with natural variation
            base_amount = recent_monthly if recent_monthly > 0 else avg_monthly_amount
            
            # Apply trend
            if trend_change > 0.2:
                base_amount *= 1.1
            elif trend_change < -0.2:
                base_amount *= 0.9
                
            # Natural monthly variation
            forecast_results[cat] = [
                base_amount * 0.9,   # Month 1
                base_amount * 1.1,   # Month 2: Higher
                base_amount * 0.95   # Month 3: Normal
            ]
            print(f"Debug: '{cat}' classified as ESSENTIAL MONTHLY")
            
        elif is_regular_monthly:
            # Regular monthly: Most months with some gaps
            base_amount = recent_monthly if recent_monthly > 0 else avg_monthly_amount
            
            # Apply trend
            if trend_change > 0.2:
                base_amount *= 1.15
            elif trend_change < -0.2:
                base_amount *= 0.85
                
            # Show in 2 out of 3 months
            forecast_results[cat] = [
                base_amount * 0.8,   # Month 1
                base_amount * 0.3,   # Month 2: Light month
                base_amount * 1.1    # Month 3: Higher
            ]
            print(f"Debug: '{cat}' classified as REGULAR MONTHLY")
            
        elif is_occasional_monthly:
            # Occasional: 1-2 out of 3 months
            base_amount = avg_monthly_amount if avg_monthly_amount > 0 else recent_monthly
            
            # More sensitive to trends for occasional expenses
            if trend_change > 0.3:
                base_amount *= 1.2
            elif trend_change < -0.3:
                base_amount *= 0.8
                
            forecast_results[cat] = [
                base_amount * 0.7,   # Month 1
                0,                   # Month 2: Skip
                base_amount * 0.9    # Month 3
            ]
            print(f"Debug: '{cat}' classified as OCCASIONAL MONTHLY")
            
        elif is_rare_monthly:
            # Rare: Maybe once in 3 months
            base_amount = avg_monthly_amount if avg_monthly_amount > 0 else 1000
            
            forecast_results[cat] = [
                0,                      # Month 1: Skip
                base_amount * 0.3,      # Month 2: Minimal
                0                       # Month 3: Skip
            ]
            print(f"Debug: '{cat}' classified as RARE MONTHLY")
            
        else:
            # Default: spread based on frequency
            if monthly_frequency > 0:
                base_amount = avg_monthly_amount
                months_to_show = max(1, int(monthly_frequency * 3))
                forecast_results[cat] = [base_amount if i < months_to_show else base_amount * 0.2 for i in range(3)]
            else:
                forecast_results[cat] = [0, 0, 0]
                
        # Clean up results
        forecast_results[cat] = [round(max(0, v), 2) for v in forecast_results[cat]]
    
    return forecast_results, months_labels


def forecast_weekly_expenses(df):
    """
    Original weekly forecast - keeping for compatibility
    """
    expenses = df[df['type'] == 'expense']
    if expenses.empty:
        return {}, []

    # Create both weekly and monthly aggregations for better pattern detection
    weekly = expenses.groupby(
        [pd.Grouper(key='date', freq='W-MON'), 'category'])['amount'].sum().reset_index()
    monthly = expenses.groupby(
        [pd.Grouper(key='date', freq='M'), 'category'])['amount'].sum().reset_index()
    
    # Get total time periods for frequency calculation
    total_weeks = len(weekly['date'].unique())
    total_months = len(monthly['date'].unique())
    categories = weekly['category'].unique()
    forecast_results = {}
    last_week = weekly['date'].max()
    weeks_labels = [(last_week + pd.Timedelta(weeks=i)
                     ).strftime('%Y-%m-%d') for i in range(1, 5)]

    print(f"Debug: Forecasting for {len(categories)} categories over {total_weeks} weeks / {total_months} months")

    for cat in categories:
        cat_weekly = weekly[weekly['category'] == cat].set_index('date').sort_index()
        cat_monthly = monthly[monthly['category'] == cat].set_index('date').sort_index()
        
        # Calculate spending patterns
        weeks_with_spending = len(cat_weekly)
        months_with_spending = len(cat_monthly)
        weekly_frequency = weeks_with_spending / total_weeks if total_weeks > 0 else 0
        monthly_frequency = months_with_spending / total_months if total_months > 0 else 0
        
        # Analyze transaction amounts to detect patterns
        avg_weekly_amount = cat_weekly['amount'].mean() if len(cat_weekly) > 0 else 0
        avg_monthly_amount = cat_monthly['amount'].mean() if len(cat_monthly) > 0 else 0
        max_transaction = expenses[expenses['category'] == cat]['amount'].max()
        
        print(f"Debug: '{cat}' - Weekly: {weeks_with_spending}/{total_weeks} ({weekly_frequency:.1%}), Monthly: {months_with_spending}/{total_months} ({monthly_frequency:.1%}), Avg: ₹{avg_weekly_amount:.0f}/week, Max: ₹{max_transaction:.0f}")
        
        # Improved categorization logic with better semantic understanding
        # Essential categories that people need regularly (with semantic improvements)
        essential_daily = ['groceries', 'commute', 'fuel', 'transport']
        essential_weekly = ['food'] if avg_weekly_amount > 0 and weekly_frequency > 0.4 else []
        monthly_bills = ['rent', 'utilities', 'insurance', 'subscription', 'mortgage', 'loan']
        
        is_monthly_expense = (monthly_frequency >= 0.8) and (weekly_frequency < 0.3) or cat.lower() in monthly_bills
        is_essential_daily = cat.lower() in essential_daily
        is_essential_weekly = cat.lower() in essential_weekly
        is_regular_expense = weekly_frequency >= 0.3 and not is_monthly_expense
        is_occasional_expense = 0.05 <= weekly_frequency < 0.3  # Lowered threshold
        is_rare_expense = weekly_frequency < 0.05  # Very rare
        
        # Detect recent trend changes (adaptability improvement)
        recent_data = cat_weekly.tail(4) if len(cat_weekly) >= 4 else cat_weekly  # Last 4 weeks
        older_data = cat_weekly.head(len(cat_weekly)-4) if len(cat_weekly) > 4 else cat_weekly
        
        recent_avg = recent_data['amount'].mean() if len(recent_data) > 0 else 0
        older_avg = older_data['amount'].mean() if len(older_data) > 0 else recent_avg
        
        # Detect if spending is increasing/decreasing (habit change detection)
        trend_change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        is_trending_up = trend_change > 0.2  # 20% increase in recent spending
        is_trending_down = trend_change < -0.2  # 20% decrease in recent spending
        
        print(f"Debug: '{cat}' - Recent avg: ₹{recent_avg:.0f}, Older avg: ₹{older_avg:.0f}, Trend: {trend_change:.1%}")
        
        # Large transaction detection (improved)
        is_large_transaction = max_transaction > avg_monthly_amount * 1.5 if avg_monthly_amount > 0 else max_transaction > 5000
        
        # Force classifications for known categories
        if cat.lower() in monthly_bills:
            is_monthly_expense = True
            is_regular_expense = False
            
        # Essential daily expenses that should appear in most weeks
        is_essential_expense = is_essential_daily or is_essential_weekly
            
        if is_monthly_expense:
            # Monthly expenses: Show once every 4 weeks with trend adjustment
            monthly_amount = avg_monthly_amount if avg_monthly_amount > 0 else cat_weekly['amount'].iloc[-1]
            
            # Apply trend changes for adaptability
            if is_trending_up:
                monthly_amount *= 1.1  # 10% increase if trending up
            elif is_trending_down:
                monthly_amount *= 0.9  # 10% decrease if trending down
                
            weekly_equivalent = monthly_amount / 4
            forecast_results[cat] = [weekly_equivalent, 0, 0, 0]
            print(f"Debug: '{cat}' classified as MONTHLY (₹{monthly_amount:.0f}/month, trend: {trend_change:.1%})")
            
        elif is_rare_expense or is_large_transaction:
            # Rare/large expenses: Show minimally but consider recent changes
            base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount
            min_amount = 200 if is_essential_expense else 50
            
            # If trending up, show slightly more; if down, show less
            multiplier = 0.3 if is_trending_up else 0.15 if is_trending_down else 0.2
            
            forecast_results[cat] = [
                min_amount,
                0,
                max(base_amount * multiplier, min_amount),
                min_amount * 0.5
            ]
            print(f"Debug: '{cat}' classified as RARE/LARGE (trend-adjusted: {multiplier})")
            
        elif is_essential_expense:
            # Essential expenses: Always present with trend adaptation
            if len(cat_weekly) > 0:
                base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount
                
                # Essential expenses adapt more conservatively to trends
                if is_trending_up:
                    base_amount *= 1.05  # 5% increase
                elif is_trending_down:
                    base_amount *= 0.95  # 5% decrease
                
                min_essential = base_amount * 0.4  # Never go below 40% of adjusted average
                forecast_results[cat] = [
                    base_amount * 0.9,
                    max(base_amount * 0.6, min_essential),
                    base_amount * 1.0,
                    max(base_amount * 0.7, min_essential)
                ]
            else:
                # Default essential spending categories
                default_essential = 3000 if cat.lower() == 'groceries' else 1500 if cat.lower() == 'food' else 1000
                forecast_results[cat] = [default_essential * 0.8, default_essential * 0.6, default_essential, default_essential * 0.7]
            
            forecast_results[cat] = [round(v, 2) for v in forecast_results[cat]]
            print(f"Debug: '{cat}' classified as ESSENTIAL (trend-adjusted)")
            
        elif is_occasional_expense:
            # Monthly expenses: Show once every 4 weeks (approximately monthly)
            monthly_amount = avg_monthly_amount if avg_monthly_amount > 0 else cat_weekly['amount'].iloc[-1]
            weekly_equivalent = monthly_amount / 4  # Spread across 4 weeks for weekly forecast
            
            forecast_results[cat] = [weekly_equivalent, 0, 0, 0]  # Show in first week only
            print(f"Debug: '{cat}' classified as MONTHLY (₹{monthly_amount:.0f}/month)")
            
        elif is_rare_expense or is_large_transaction:
            # Rare/large expenses: Show minimally with reduced frequency
            avg_amount = cat_weekly['amount'].mean() if len(cat_weekly) > 0 else 0
            forecast_results[cat] = [0, 0, avg_amount * 0.2, 0]  # Minimal forecast in week 3
            print(f"Debug: '{cat}' classified as RARE/LARGE")
            
        elif is_occasional_expense:
            # Occasional expenses: Show in 1-2 weeks with trend adaptation
            base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount
            
            # Occasional expenses adapt more to trend changes
            if is_trending_up:
                base_amount *= 1.15  # 15% increase
            elif is_trending_down:
                base_amount *= 0.85  # 15% decrease
            
            # Show in weeks 1 and 3 with variation (avoid complete zeros)
            forecast_results[cat] = [
                base_amount * 0.9,    # Week 1: Near full amount
                base_amount * 0.3,    # Week 2: Minimal spending
                base_amount * 0.8,    # Week 3: Reduced amount
                base_amount * 0.4     # Week 4: Light spending
            ]
            print(f"Debug: '{cat}' classified as OCCASIONAL (trend-adjusted: {trend_change:.1%})")
            
        elif is_regular_expense:
            # Regular expenses: Use trend-adapted forecasting
            if len(cat_weekly) < 3:
                # Apply variation for regular expenses with limited data
                if len(cat_weekly) >= 1:
                    base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount
                    
                    # Apply trend adjustment
                    if is_trending_up:
                        base_amount *= 1.1
                    elif is_trending_down:
                        base_amount *= 0.9
                    
                    # Create realistic weekly variation (avoid zeros for regular expenses)
                    forecast_results[cat] = [
                        base_amount * 0.95,   # Week 1
                        base_amount * 0.7,    # Week 2: Light week
                        base_amount * 1.05,   # Week 3: Above average
                        base_amount * 0.8     # Week 4: Below average
                    ]
                else:
                    # Even for no data, show minimal realistic spending
                    estimate = 2000 if cat.lower() in ['groceries', 'shopping'] else 800
                    forecast_results[cat] = [estimate * 0.8, estimate * 0.5, estimate, estimate * 0.6]
                print(f"Debug: '{cat}' classified as REGULAR (sparse data, trend: {trend_change:.1%})")
                continue
                
            # Use exponential smoothing for regular expenses with sufficient data
            try:
                model = ExponentialSmoothing(
                    cat_weekly['amount'], trend='add', seasonal=None, initialization_method="estimated").fit()
                forecast = model.forecast(4)
                
                # Apply trend adjustment to forecasted values
                trend_multiplier = 1.0
                if is_trending_up:
                    trend_multiplier = 1.1
                elif is_trending_down:
                    trend_multiplier = 0.9
                
                # Add realistic variation but avoid zeros
                forecast_with_variation = [
                    forecast[0] * trend_multiplier,
                    max(forecast[1] * 0.5 * trend_multiplier, forecast[1] * 0.3),  # Light week
                    forecast[2] * 0.9 * trend_multiplier,
                    max(forecast[3] * 0.6 * trend_multiplier, forecast[3] * 0.4)   # Another light week
                ]
                forecast_results[cat] = [round(max(200, v), 2) for v in forecast_with_variation]  # Minimum ₹200
                print(f"Debug: '{cat}' classified as REGULAR (exponential smoothing, trend: {trend_change:.1%})")
            except Exception as e:
                print(f"Debug: Exponential smoothing failed for {cat}: {e}")
                # Fallback with light weeks but no zeros
                base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount
                if is_trending_up:
                    base_amount *= 1.1
                elif is_trending_down:
                    base_amount *= 0.9
                    
                forecast_results[cat] = [
                    base_amount * 0.95, 
                    base_amount * 0.5,  # Light week
                    base_amount * 1.0, 
                    base_amount * 0.6   # Another light week
                ]
                forecast_results[cat] = [round(max(200, v), 2) for v in forecast_results[cat]]
                print(f"Debug: '{cat}' classified as REGULAR (fallback, trend: {trend_change:.1%})")
        else:
            # Default case
            forecast_results[cat] = [0] * 4
            print(f"Debug: '{cat}' - no clear pattern, zero forecast")
    
    return forecast_results, weeks_labels


def clean_forecast_results(forecast_results):
    """Remove categories with minimal or no predicted spending to declutter the forecast"""
    cleaned_results = {}
    
    for category, values in forecast_results.items():
        # Calculate total predicted spending for this category
        total_predicted = sum(values)
        max_week_spending = max(values) if values else 0
        
        # More lenient filtering - keep categories that have any meaningful prediction
        # Essential categories (food, groceries, etc.) should always be kept
        is_essential = category.lower() in ['food', 'groceries', 'transport', 'fuel', 'commute', 'rent', 'utilities']
        
        if is_essential or total_predicted > 50 or max_week_spending > 25:  # Much lower thresholds
            cleaned_results[category] = values
            print(f"Debug: Keeping category '{category}' (total: {total_predicted}, max: {max_week_spending}, essential: {is_essential})")
        else:
            print(f"Debug: Filtering out category '{category}' (total: {total_predicted}, max: {max_week_spending})")
    
    return cleaned_results


def forecast_net_cashflow(df):
    if df.empty:
        return [], []
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    df_copy['net'] = np.where(df_copy['type'] == 'revenue', df_copy['amount'], -df_copy['amount'])
    weekly_net = df_copy.groupby(pd.Grouper(
        key='date', freq='W-MON'))['net'].sum().sort_index()
    
    print(f"Debug: Cashflow has {len(weekly_net)} weekly data points")
    
    if len(weekly_net) < 3:
        if len(weekly_net) > 0:
            last_net = weekly_net.iloc[-1]
            
            # Add some variation for sparse cashflow data
            labels = [(weekly_net.index.max() + pd.Timedelta(weeks=i)
                       ).strftime('%Y-%m-%d') for i in range(1, 5)]
            
            # Create slight variation instead of flat repetition
            if len(weekly_net) >= 2:
                trend = weekly_net.iloc[-1] - weekly_net.iloc[0]
                trend_per_week = trend / len(weekly_net)
            else:
                trend_per_week = 0
            
            values = []
            for i in range(4):
                val = last_net + (trend_per_week * (i + 1) * 0.2)  # Small trend continuation
                # Add alternating variation to avoid completely flat forecast
                variation = abs(last_net) * 0.02 * (1 if i % 2 == 0 else -1)
                values.append(round(val + variation, 2))
            
            print(f"Debug: Used trend-based cashflow forecast: {values}")
        else:
            labels = []
            values = [0] * 4
    else:
        try:
            model = ExponentialSmoothing(
                weekly_net, trend='add', seasonal=None, initialization_method="estimated").fit()
            forecast = model.forecast(4)
            labels = [(weekly_net.index.max() + pd.Timedelta(weeks=i)
                       ).strftime('%Y-%m-%d') for i in range(1, 5)]
            values = forecast.round(2).tolist()
            print(f"Debug: Used exponential smoothing for cashflow: {values}")
        except Exception as e:
            print(f"Debug: Cashflow exponential smoothing failed, using fallback: {e}")
            # Fallback to trend-based forecast
            recent_avg = weekly_net.iloc[-2:].mean()
            older_avg = weekly_net.iloc[:-2].mean() if len(weekly_net) > 2 else recent_avg
            trend = recent_avg - older_avg
            
            labels = [(weekly_net.index.max() + pd.Timedelta(weeks=i)
                       ).strftime('%Y-%m-%d') for i in range(1, 5)]
            values = [round(recent_avg + trend * (i + 1) * 0.3, 2) for i in range(4)]
            
    return labels, values


def generate_insights(df):
    insights = []
    warnings = []
    if df.empty:
        return insights, warnings
    now = pd.Timestamp.now(tz='UTC')
    this_month = df[df['date'].dt.month == now.month]
    last_3_months = df[(df['date'] >= now -
                        pd.DateOffset(months=3)) & (df['date'] < now)]

    revenue = this_month[this_month['type'] == 'revenue']['amount'].sum()
    expense = this_month[this_month['type'] == 'expense']['amount'].sum()
    savings = revenue - expense
    savings_rate = (savings / revenue) if revenue > 0 else 0

    if savings_rate < 0.3:
        insights.append(
            f"Your savings rate this month is {int(savings_rate*100)}%. Try to increase it to at least 40%.")

    cat_avg = last_3_months[last_3_months['type'] == 'expense'].groupby('category')[
        'amount'].mean()
    cat_this = this_month[this_month['type'] == 'expense'].groupby('category')[
        'amount'].sum()
    for cat in cat_this.index:
        avg = cat_avg.get(cat, 0)
        curr = cat_this[cat]
        if avg == 0:
            continue
        if curr > 1.5 * avg:
            diff = int(curr - avg)
            insights.append(
                f"Your expenses in {cat} are {int(((curr/avg)-1)*100)}% above your average. Consider cutting down to save Rs {diff} this month.")

    revenue_by_source = this_month[this_month['type'] == 'revenue'].groupby('category')[
        'amount'].sum()
    total_revenue = revenue_by_source.sum()
    freelance_share = (revenue_by_source.get('Freelance', 0) /
                       total_revenue) if total_revenue > 0 else 0
    if freelance_share > 0.2:
        insights.append(
            f"Your income relies {int(freelance_share*100)}% on freelance work. Plan for unpredictability.")

    expense_last_month = last_3_months[last_3_months['type']
                                       == 'expense']['amount'].sum()
    expense_this_month = expense
    if expense_this_month > expense_last_month * 1.2:
        warnings.append(
            "Your expenses increased sharply this month, you might save less next month if this continues.")

    # Emergency fund insight removed as per user request

    if savings > 0:
        months = 12 * 5
        rate = 0.12 / 12
        fv = savings * (((1 + rate) ** months - 1) / rate) * (1 + rate)
        insights.append(
            f"If you invest Rs {int(savings)}/month in a 12% SIP, you will have approximately Rs {int(fv)} in 5 years.")

    # Additional Warning Conditions

    # 1. Low Savings Rate Warning
    if savings_rate < 0.1 and revenue > 0:
        warnings.append(
            f"Your savings rate is only {int(savings_rate*100)}%. You're saving very little - consider reducing expenses or increasing income.")

    # 2. Negative Savings Warning
    if savings < 0:
        deficit = int(abs(savings))
        warnings.append(
            f"You're spending Rs {deficit} more than you earn this month. This is unsustainable!")

    # 3. High Single Category Spending Warning
    for cat in cat_this.index:
        cat_percentage = (cat_this[cat] / expense) * 100 if expense > 0 else 0
        if cat_percentage > 40:  # If any category is more than 40% of total expenses
            warnings.append(
                f"Your {cat} expenses are {int(cat_percentage)}% of your total spending. Consider diversifying your budget.")

    # 4. No Income Warning
    if revenue == 0:
        warnings.append(
            "No income recorded this month. Make sure to track all your income sources.")

    # 5. Excessive Luxury Spending Warning
    luxury_categories = ['shopping', 'entertainment', 'dining', 'travel']
    luxury_spending = this_month[
        (this_month['type'] == 'expense') &
        (this_month['category'].isin(luxury_categories))
    ]['amount'].sum()

    if expense > 0 and (luxury_spending / expense) > 0.3:
        luxury_percentage = int((luxury_spending / expense) * 100)
        warnings.append(
            f"You're spending {luxury_percentage}% on luxury items (shopping, entertainment, dining, travel). Consider reducing these for better savings.")

    # 6. Irregular Income Warning
    if len(revenue_by_source) == 1 and total_revenue > 0:
        single_source = revenue_by_source.index[0]
        warnings.append(
            f"All your income comes from {single_source}. Consider diversifying your income sources for financial stability.")

    # 7. High Monthly Expense Growth Warning
    if len(last_3_months) > 0:
        monthly_expenses = last_3_months[last_3_months['type'] == 'expense'].groupby(
            pd.Grouper(key='date', freq='M'))['amount'].sum()
        if len(monthly_expenses) >= 2:
            last_month_expense = monthly_expenses.iloc[-1]
            prev_month_expense = monthly_expenses.iloc[-2]
            if prev_month_expense > 0 and (last_month_expense - prev_month_expense) / prev_month_expense > 0.15:
                increase_percentage = int(
                    ((last_month_expense - prev_month_expense) / prev_month_expense) * 100)
                warnings.append(
                    f"Your monthly expenses increased by {increase_percentage}% last month. Monitor your spending carefully.")

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


@app.get("/api/debug/classification/{userId}")
async def debug_classification(userId: str):
    """Debug endpoint to show how categories are classified based on spending patterns."""
    try:
        data = fetch_data(user_id=userId, months=6)
        df = prepare_dataframe(data)
        
        if df.empty:
            return {"error": "No data found"}
        
        expenses = df[df['type'] == 'expense']
        weekly = expenses.groupby(
            [pd.Grouper(key='date', freq='W-MON'), 'category'])['amount'].sum().reset_index()
        monthly = expenses.groupby(
            [pd.Grouper(key='date', freq='M'), 'category'])['amount'].sum().reset_index()
        
        total_weeks = len(weekly['date'].unique())
        total_months = len(monthly['date'].unique())
        categories = weekly['category'].unique()
        
        classification_results = {}
        
        for cat in categories:
            cat_weekly = weekly[weekly['category'] == cat].set_index('date').sort_index()
            cat_monthly = monthly[monthly['category'] == cat].set_index('date').sort_index()
            
            weeks_with_spending = len(cat_weekly)
            months_with_spending = len(cat_monthly)
            weekly_frequency = weeks_with_spending / total_weeks if total_weeks > 0 else 0
            monthly_frequency = months_with_spending / total_months if total_months > 0 else 0
            
            avg_weekly_amount = cat_weekly['amount'].mean() if len(cat_weekly) > 0 else 0
            avg_monthly_amount = cat_monthly['amount'].mean() if len(cat_monthly) > 0 else 0
            max_transaction = expenses[expenses['category'] == cat]['amount'].max()
            
            # Apply same classification logic
            is_monthly_expense = (monthly_frequency >= 0.8) and (weekly_frequency < 0.3)
            is_regular_expense = weekly_frequency >= 0.3
            is_occasional_expense = 0.1 <= weekly_frequency < 0.3
            is_rare_expense = weekly_frequency < 0.1
            is_large_transaction = max_transaction > avg_monthly_amount * 2 if avg_monthly_amount > 0 else False
            is_essential_expense = cat.lower() in ['food', 'groceries', 'transport', 'fuel', 'commute']
            
            if cat.lower() in ['rent', 'utilities', 'insurance']:
                is_monthly_expense = True
                is_regular_expense = False
            
            # Determine final classification
            if is_monthly_expense:
                classification = "MONTHLY"
            elif is_rare_expense or is_large_transaction:
                classification = "RARE/LARGE"
            elif is_essential_expense:
                classification = "ESSENTIAL"
            elif is_occasional_expense:
                classification = "OCCASIONAL"
            elif is_regular_expense:
                classification = "REGULAR"
            else:
                classification = "UNKNOWN"
            
            classification_results[cat] = {
                "classification": classification,
                "weekly_frequency": f"{weekly_frequency:.1%}",
                "monthly_frequency": f"{monthly_frequency:.1%}",
                "weeks_with_spending": f"{weeks_with_spending}/{total_weeks}",
                "months_with_spending": f"{months_with_spending}/{total_months}",
                "avg_weekly_amount": round(avg_weekly_amount, 2),
                "avg_monthly_amount": round(avg_monthly_amount, 2),
                "max_transaction": round(max_transaction, 2),
                "is_essential": is_essential_expense,
                "is_large_transaction": is_large_transaction
            }
        
        return {
            "total_periods": f"{total_weeks} weeks, {total_months} months",
            "categories": classification_results
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/collections")
async def debug_collections():
    """Debug endpoint to check what's in the collections."""
    try:
        # Get sample users
        sample_users = list(db['users'].find(
            {}, {"_id": 1, "email": 1}).limit(5))

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
        expense_forecast, expense_labels = forecast_monthly_expenses(df)
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

        forecast_data, forecast_labels = forecast_monthly_expenses(df)
        # Clean the forecast data to remove categories with minimal spending
        cleaned_forecast_data = clean_forecast_results(forecast_data)
        cashflow_labels, cashflow_values = forecast_net_cashflow(df)
        insights, warnings = generate_insights(df)

        return {
            "forecast_chart_data": {"labels": forecast_labels, "categories": cleaned_forecast_data},
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
                weekly_labels.append(
                    (today + timedelta(weeks=i)).strftime('%Y-%m-%d'))

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

        forecast_data, forecast_labels = forecast_monthly_expenses(df)
        # Clean the forecast data to remove categories with minimal spending
        cleaned_forecast_data = clean_forecast_results(forecast_data)
        cashflow_labels, cashflow_values = forecast_net_cashflow(df)
        insights, warnings = generate_insights(df)

        return {
            "forecast_chart_data": {"labels": forecast_labels, "categories": cleaned_forecast_data},
            "cashflow_chart_data": {"labels": cashflow_labels, "values": cashflow_values},
            "insights": insights,
            "warnings": warnings
        }
    except Exception as e:
        print(f"Error in api_forecast_post: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
