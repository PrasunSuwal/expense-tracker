import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# MongoDB Setup
# =====================================================
client = MongoClient(
    "mongodb+srv://dishantghimire10:3hsQ2m4JAJlNKDO8@expensetracker.87kcsh4.mongodb.net/?appName=expenseTracker"
)
db = client['test']

# =====================================================
# User ID (Replace with your test user ID)
# =====================================================
USER_ID = "690b2955bdfdafcf3fa700ee"  # Replace with actual user ID from your database

# =====================================================
# Helper Functions (from forecast_api.py)
# =====================================================

def fetch_data(user_id: str, months=12):
    """Fetch expense data from MongoDB"""
    try:
        from bson import ObjectId
        user_obj_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error converting userId to ObjectId: {e}")
        return []

    date_cutoff = datetime.utcnow() - relativedelta(months=months)
    query = {
        "userId": user_obj_id,
        "date": {"$gte": date_cutoff}
    }
    
    # Fetch expenses only (matching your forecast model)
    expense_cursor = db['expenses'].find(query)
    expense_data = []
    
    for item in expense_cursor:
        try:
            if "amount" in item and item["amount"] > 0:
                date = item["date"]
                if isinstance(date, str):
                    try:
                        date = datetime.strptime(date.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z")
                
                if date.year < 1900 or date.year > 2100:
                    continue
                
                expense_data.append({
                    "date": date,
                    "amount": float(item["amount"]),
                    "type": "expense",
                    "category": item.get("category", "Other"),
                    "userId": str(item["userId"]),
                    "created": item.get("createdAt"),
                    "updated": item.get("updatedAt")
                })
        except Exception as e:
            print(f"Error processing expense record: {e}")
    
    return expense_data


def prepare_dataframe(data):
    """Prepare DataFrame from raw data"""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    for date_field in ['date', 'created', 'updated']:
        if date_field in df.columns:
            df[date_field] = pd.to_datetime(df[date_field], utc=True)
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    
    return df


def generate_synthetic_data(df, target_months=12):
    """Generate synthetic data to reach target_months"""
    if df.empty:
        return df
    
    df_sorted = df.sort_values('date')
    date_range = (df_sorted['date'].max() - df_sorted['date'].min()).days / 30.44
    current_months = int(date_range) + 1
    
    if current_months >= target_months:
        return df
    
    months_needed = target_months - current_months
    synthetic_records = []
    earliest_date = df['date'].min()
    
    for cat in df['category'].unique():
        for trans_type in df['type'].unique():
            cat_type_data = df[(df['category'] == cat) & (df['type'] == trans_type)]
            
            if len(cat_type_data) == 0:
                continue
            
            cat_monthly = cat_type_data.groupby(cat_type_data['date'].dt.to_period('M'))['amount'].sum()
            
            if len(cat_monthly) < 2:
                avg_amount = cat_type_data['amount'].mean()
                growth_rate = 0
            else:
                amounts = cat_monthly.values
                avg_amount = np.mean(amounts)
                growth_rate = (amounts[-1] - amounts[0]) / len(amounts) if len(amounts) > 1 else 0
            
            months_span = len(cat_monthly)
            transaction_count = len(cat_type_data)
            transactions_per_month = transaction_count / months_span if months_span > 0 else 1
            
            for month_offset in range(1, months_needed + 1):
                synthetic_date = earliest_date - relativedelta(months=month_offset)
                synthetic_amount = avg_amount - (growth_rate * month_offset)
                synthetic_amount = max(synthetic_amount, avg_amount * 0.5)
                
                num_transactions = max(1, int(transactions_per_month))
                
                for i in range(num_transactions):
                    day_offset = int(30 * (i / num_transactions))
                    transaction_date = synthetic_date + timedelta(days=day_offset)
                    
                    variation = np.random.uniform(0.8, 1.2)
                    amount = synthetic_amount * variation / num_transactions
                    
                    synthetic_records.append({
                        'date': transaction_date,
                        'amount': round(amount, 2),
                        'type': trans_type,
                        'category': cat,
                        'userId': df['userId'].iloc[0],
                        'created': transaction_date,
                        'updated': transaction_date,
                        'is_synthetic': True
                    })
    
    synthetic_df = pd.DataFrame(synthetic_records)
    if not synthetic_df.empty:
        df['is_synthetic'] = False
        combined_df = pd.concat([synthetic_df, df], ignore_index=True).sort_values('date')
        return combined_df
    
    return df


# =====================================================
# Model Evaluation
# =====================================================

def evaluate_xgboost_model():
    """
    Evaluate the XGBoost model exactly as implemented in forecast_api.py
    """
    print("=" * 80)
    print("XGBoost MODEL EVALUATION REPORT")
    print("=" * 80)
    print()
    
    # Fetch data
    print(f"Fetching data for user: {USER_ID}")
    data = fetch_data(user_id=USER_ID, months=12)
    
    if not data:
        print("❌ No data found for this user.")
        return
    
    df = prepare_dataframe(data)
    print(f"✓ Fetched {len(df)} expense records")
    print(f"✓ Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # Add synthetic data if needed
    df_with_synthetic = generate_synthetic_data(df, target_months=12)
    expenses = df_with_synthetic[df_with_synthetic['type'] == 'expense'].copy()
    
    print(f"✓ Total records after synthetic data: {len(expenses)}")
    if 'is_synthetic' in expenses.columns:
        synthetic_count = expenses['is_synthetic'].sum()
        print(f"✓ Synthetic records added: {synthetic_count}")
    print()
    
    # Normalize category names
    expenses['category'] = expenses['category'].str.strip().str.capitalize()
    
    # Group by month and category
    monthly_data = expenses.groupby([pd.Grouper(key='date', freq='ME'), 'category'])['amount'].sum().reset_index()
    monthly_data.rename(columns={'date': 'month_end'}, inplace=True)
    monthly_data['date'] = monthly_data['month_end']
    monthly_data['type'] = 'expense'
    
    categories = monthly_data['category'].unique()
    print(f"✓ Evaluating {len(categories)} expense categories")
    print()
    
    # Evaluate each category
    all_predictions = []
    all_actuals = []
    category_results = []
    
    for cat in categories:
        cat_data = monthly_data[monthly_data['category'] == cat].copy()
        
        if len(cat_data) < 4:
            continue
        
        # Prepare features (same as forecast_api.py)
        cat_data = cat_data.sort_values('date').reset_index(drop=True)
        cat_data['month_num'] = range(len(cat_data))
        cat_data['lag_1'] = cat_data['amount'].shift(1).fillna(cat_data['amount'].mean())
        cat_data['lag_2'] = cat_data['amount'].shift(2).fillna(cat_data['amount'].mean())
        cat_data['rolling_mean_3'] = cat_data['amount'].rolling(window=3, min_periods=1).mean()
        
        X = cat_data[['month_num', 'lag_1', 'lag_2', 'rolling_mean_3']].values
        y = cat_data['amount'].values
        
        # Train-test split (80-20)
        if len(X) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train XGBoost model (same parameters as forecast_api.py)
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=30,
            max_depth=3,
            learning_rate=0.15,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store results
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        
        # Calculate metrics for this category
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate confidence (same as forecast_api.py)
        recent_3 = cat_data['amount'].tail(3)
        variance = recent_3.std() / recent_3.mean() if recent_3.mean() > 0 else 0
        cat_confidence = max(70, min(95, 95 - (variance * 50)))
        
        category_results.append({
            'category': cat,
            'data_points': len(cat_data),
            'test_size': len(y_test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'confidence': cat_confidence,
            'avg_actual': np.mean(y_test),
            'avg_predicted': np.mean(y_pred)
        })
    
    # Overall metrics
    if all_predictions and all_actuals:
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        overall_r2 = r2_score(all_actuals, all_predictions)
        overall_confidence = np.mean([r['confidence'] for r in category_results])
        
        print("=" * 80)
        print("OVERALL MODEL PERFORMANCE")
        print("=" * 80)
        print(f"Total Categories Evaluated: {len(category_results)}")
        print(f"Total Test Predictions: {len(all_predictions)}")
        print()
        print(f"Mean Absolute Error (MAE):  Rs. {overall_mae:.2f}")
        print(f"Root Mean Squared Error:     Rs. {overall_rmse:.2f}")
        print(f"R² Score:                    {overall_r2:.3f}")
        print(f"Overall Confidence:          {overall_confidence:.1f}%")
        print()
        
        # Confidence level interpretation
        if overall_r2 >= 0.85:
            conf_level = "HIGH ✓✓✓"
        elif overall_r2 >= 0.6:
            conf_level = "MEDIUM ✓✓"
        else:
            conf_level = "LOW ✓"
        
        print(f"Confidence Level: {conf_level}")
        print()
        
        # Category breakdown
        print("=" * 80)
        print("CATEGORY-WISE PERFORMANCE")
        print("=" * 80)
        print(f"{'Category':<20} {'Data Points':<12} {'MAE':<12} {'R²':<10} {'Confidence':<12}")
        print("-" * 80)
        
        for result in sorted(category_results, key=lambda x: x['r2'], reverse=True):
            print(f"{result['category']:<20} {result['data_points']:<12} Rs. {result['mae']:<8.0f} {result['r2']:<10.3f} {result['confidence']:<10.1f}%")
        
        print()
        
        # Sample predictions vs actuals
        print("=" * 80)
        print("SAMPLE PREDICTIONS vs ACTUALS (Last 10 test points)")
        print("=" * 80)
        print(f"{'Actual':<15} {'Predicted':<15} {'Error':<15} {'Error %':<15}")
        print("-" * 80)
        
        sample_size = min(10, len(all_actuals))
        for i in range(-sample_size, 0):
            actual = all_actuals[i]
            predicted = all_predictions[i]
            error = abs(actual - predicted)
            error_pct = (error / actual * 100) if actual > 0 else 0
            
            print(f"Rs. {actual:<11.0f} Rs. {predicted:<11.0f} Rs. {error:<11.0f} {error_pct:<13.1f}%")
        
        print()
        print("=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        # Recommendations
        print("\n📊 Model Quality Assessment:")
        if overall_r2 >= 0.85:
            print("✅ Excellent model performance! Predictions are highly reliable.")
        elif overall_r2 >= 0.7:
            print("✓ Good model performance. Predictions are generally reliable.")
        elif overall_r2 >= 0.5:
            print("⚠️  Moderate performance. Consider adding more historical data.")
        else:
            print("❌ Low performance. More consistent transaction data needed.")
        
        print(f"\n💡 Tip: Model accuracy improves with more consistent expense tracking.")
        print(f"   Current data spans {len(monthly_data['date'].unique())} months.")
        print(f"   Aim for at least 12 months of consistent data for best results.")
    
    else:
        print("❌ Not enough data to evaluate model.")


# =====================================================
# Run Evaluation
# =====================================================

if __name__ == "__main__":
    try:
        evaluate_xgboost_model()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
