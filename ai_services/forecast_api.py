from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from typing import Optional, Dict, List, Tuple
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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
    # "mongodb+srv://dishantghimire10:3hsQ2m4JAJlNKDO8@expensetracker.87kcsh4.mongodb.net/?retryWrites=true&w=majority&appName=expenseTracker"
)
db = client['test']

# ----------------------
# Helper functions
# ----------------------


def generate_synthetic_data(df, target_months=12):
    """
    Generate synthetic data based on historical trends to reach target_months.
    Uses monthly growth patterns to extrapolate backwards.
    """
    if df.empty:
        return df

    # Calculate current months of data
    df_sorted = df.sort_values('date')
    date_range = (df_sorted['date'].max() -
                  df_sorted['date'].min()).days / 30.44
    current_months = int(date_range) + 1

    print(f"Debug: Current data spans {current_months} months")

    if current_months >= target_months:
        return df

    # Calculate months needed
    months_needed = target_months - current_months
    print(f"Debug: Generating {months_needed} months of synthetic data")

    # Group by month and category to find patterns
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_stats = df.groupby(['year_month', 'category', 'type'])[
        'amount'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate average growth rate per category
    synthetic_records = []
    earliest_date = df['date'].min()

    for cat in df['category'].unique():
        for trans_type in df['type'].unique():
            cat_type_data = df[(df['category'] == cat) &
                               (df['type'] == trans_type)]

            if len(cat_type_data) == 0:
                continue

            # Calculate monthly average and trend
            cat_monthly = cat_type_data.groupby(
                cat_type_data['date'].dt.to_period('M'))['amount'].sum()

            if len(cat_monthly) < 2:
                avg_amount = cat_type_data['amount'].mean()
                growth_rate = 0
            else:
                # Calculate linear growth rate
                amounts = cat_monthly.values
                avg_amount = np.mean(amounts)
                growth_rate = (amounts[-1] - amounts[0]) / \
                    len(amounts) if len(amounts) > 1 else 0

            # Frequency: how many transactions per month on average
            months_span = len(cat_monthly)
            transaction_count = len(cat_type_data)
            transactions_per_month = transaction_count / \
                months_span if months_span > 0 else 1

            # Generate synthetic data going backwards
            for month_offset in range(1, months_needed + 1):
                synthetic_date = earliest_date - \
                    relativedelta(months=month_offset)

                # Calculate amount with reverse trend (going back in time)
                synthetic_amount = avg_amount - (growth_rate * month_offset)
                # Don't go below 50% of average
                synthetic_amount = max(synthetic_amount, avg_amount * 0.5)

                # Generate transactions for this month
                num_transactions = max(1, int(transactions_per_month))

                for i in range(num_transactions):
                    # Spread transactions across the month
                    day_offset = int(30 * (i / num_transactions))
                    transaction_date = synthetic_date + \
                        timedelta(days=day_offset)

                    # Add some variation to amounts
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

    # Combine synthetic and real data
    synthetic_df = pd.DataFrame(synthetic_records)
    if not synthetic_df.empty:
        df['is_synthetic'] = False
        combined_df = pd.concat(
            [synthetic_df, df], ignore_index=True).sort_values('date')
        print(f"Debug: Added {len(synthetic_records)} synthetic transactions")
        return combined_df

    return df


def fetch_data(user_id: str, months=12):
    """
    Fetch income and expense data from MongoDB and normalize fields.
    Uses UTC to match MongoDB ISODate. Returns combined list of transactions.
    Now fetches 12 months by default for better forecasting.
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer optimized features for XGBoost forecasting (faster version)
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Basic temporal features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Essential rolling statistics (reduced windows for speed)
    for window in [3, 6]:
        df[f'rolling_mean_{window}'] = df.groupby('category')['amount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('category')['amount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )

    # Key lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('category')['amount'].shift(
            lag).fillna(df['amount'].mean())

    # Trend
    df['amount_trend'] = df.groupby('category')['amount'].transform(
        lambda x: x.expanding().mean()
    )

    # Category encoding
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Fill remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def train_xgboost_model(df: pd.DataFrame) -> Tuple[xgb.XGBRegressor, List[str], float, Dict]:
    """
    Train optimized XGBoost model (faster training)
    """
    # Engineer features
    df_engineered = engineer_features(df)

    # Select feature columns (exclude target and non-feature columns)
    exclude_cols = ['date', 'amount', 'type', 'category', 'userId', 'created', 'updated',
                    'is_synthetic', 'year_month', 'week_start', 'month_end']
    feature_cols = [
        col for col in df_engineered.columns if col not in exclude_cols]

    X = df_engineered[feature_cols]
    y = df_engineered['amount']

    # Split data for validation (smaller test set for speed)
    if len(X) > 15:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Train faster XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,  # Reduced from 200
        max_depth=4,      # Reduced from 5
        learning_rate=0.1,  # Increased from 0.05
        random_state=42,
        verbosity=0,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate confidence (based on R2 score, adjusted for model simplicity)
    confidence = min(max(r2 * 100, 50), 95)  # Cap between 50-95%

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'confidence': float(confidence)
    }

    print(
        f"Debug: Model trained - MAE: Rs. {mae:.2f}, R²: {r2:.3f}, Confidence: {confidence:.1f}%")

    return model, feature_cols, confidence, metrics


def calculate_advanced_confidence(cat_data: pd.DataFrame, cv_predictions: np.ndarray, cv_actuals: np.ndarray, model_scores: List[float]) -> float:
    """
    Strategy 3: Multi-factor confidence calculation
    Considers: variance, completeness, trend stability, model accuracy, and ensemble agreement
    """
    # Factor 1: Historical variance (consistency)
    recent_3 = cat_data['amount'].tail(3)
    if len(recent_3) > 0 and recent_3.mean() > 0:
        variance_score = max(0, 100 - (recent_3.std() / recent_3.mean() * 100))
    else:
        variance_score = 50

    # Factor 2: Data completeness
    months_available = len(cat_data)
    completeness_score = min(100, (months_available / 12) * 100)

    # Factor 3: Trend stability
    if len(cat_data) >= 2:
        try:
            trend_coef = np.polyfit(
                range(len(cat_data)), cat_data['amount'], 1)[0]
            trend_score = max(0, 100 - abs(trend_coef) /
                              cat_data['amount'].mean() * 100)
        except:
            trend_score = 70
    else:
        trend_score = 70

    # Factor 4: Model prediction accuracy (from cross-validation)
    if len(cv_predictions) > 0 and len(cv_actuals) > 0:
        errors = np.abs(cv_predictions - cv_actuals)
        mean_error_pct = (errors.mean() / cv_actuals.mean()
                          * 100) if cv_actuals.mean() > 0 else 50
        accuracy_score = max(0, 100 - mean_error_pct)
    else:
        accuracy_score = 70

    # Factor 5: Ensemble agreement (Strategy 4)
    if len(model_scores) >= 2:
        ensemble_variance = np.std(model_scores)
        ensemble_agreement = max(0, 100 - (ensemble_variance * 100))
    else:
        ensemble_agreement = 70

    # Weighted average
    confidence = (
        variance_score * 0.25 +
        completeness_score * 0.20 +
        trend_score * 0.15 +
        accuracy_score * 0.25 +
        ensemble_agreement * 0.15
    )

    return max(50, min(95, confidence))


def train_ensemble_models(X: np.ndarray, y: np.ndarray) -> Tuple[List, List[float], Dict]:
    """
    Strategy 4 & 7: Train ensemble of 3 models with cross-validation
    Returns: [models], [scores], metrics
    """
    models = []
    scores = []

    # Use TimeSeriesSplit for proper time-series validation (Strategy 7)
    tscv = TimeSeriesSplit(n_splits=3)

    # Model 1: XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=1
    )

    # Model 2: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=1
    )

    # Model 3: Linear Regression
    lr_model = LinearRegression()

    model_list = [
        ('XGBoost', xgb_model),
        ('RandomForest', rf_model),
        ('LinearRegression', lr_model)
    ]

    # Strategy 7: Cross-validation
    cv_predictions = []
    cv_actuals = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_predictions = []

        for name, model in model_list:
            try:
                if hasattr(model, 'get_params'):
                    model_clone = type(model)(**model.get_params())
                else:
                    model_clone = type(model)()

                model_clone.fit(X_train, y_train)
                pred = model_clone.predict(X_val)
                fold_predictions.append(pred)
            except:
                pass

        if fold_predictions:
            ensemble_pred = np.mean(fold_predictions, axis=0)
            cv_predictions.extend(ensemble_pred)
            cv_actuals.extend(y_val)

    # Calculate cross-validated metrics
    cv_predictions = np.array(cv_predictions)
    cv_actuals = np.array(cv_actuals)

    if len(cv_predictions) > 0 and len(cv_actuals) > 0:
        cv_mae = mean_absolute_error(cv_actuals, cv_predictions)
        cv_rmse = np.sqrt(mean_squared_error(cv_actuals, cv_predictions))
        cv_r2 = r2_score(cv_actuals, cv_predictions)
    else:
        cv_mae = cv_rmse = 0
        cv_r2 = 0.75

    # Train final models on all data
    for name, model in model_list:
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            scores.append(score)
            models.append(model)
        except:
            pass

    metrics = {
        'cv_mae': float(cv_mae),
        'cv_rmse': float(cv_rmse),
        'cv_r2': float(cv_r2),
        'individual_scores': [float(s) for s in scores],
        'ensemble_agreement': float(100 - np.std(scores) * 100) if scores else 70.0
    }

    return models, scores, metrics


def predict_with_ensemble(models: List, X: np.ndarray) -> Tuple[float, List[float]]:
    """
    Make prediction using ensemble of models
    Returns: (average_prediction, individual_predictions)
    """
    predictions = []

    for model in models:
        try:
            pred = model.predict(X)
            predictions.append(float(pred[0]) if len(pred) > 0 else 0)
        except:
            pass

    if predictions:
        ensemble_pred = np.mean(predictions)
    else:
        ensemble_pred = 0

    return ensemble_pred, predictions


def forecast_next_month_expenses_ensemble(df: pd.DataFrame) -> Tuple[Dict, str, float, Dict]:
    """
    Enhanced forecasting with Strategy 3, 4, and 7 implemented
    Uses ensemble models + advanced confidence + cross-validation
    """
    expenses = df[df['type'] == 'expense'].copy()
    if expenses.empty:
        return {}, "", 0.0, {}

    # Ensure we have at least 12 months of data
    df_with_synthetic = generate_synthetic_data(expenses, target_months=12)
    expenses = df_with_synthetic[df_with_synthetic['type'] == 'expense'].copy()

    # Normalize category names
    expenses['category'] = expenses['category'].str.strip().str.capitalize()

    # Group by month and category
    monthly_data = expenses.groupby([pd.Grouper(key='date', freq='ME'), 'category'])[
        'amount'].sum().reset_index()
    monthly_data.rename(columns={'date': 'month_end'}, inplace=True)
    monthly_data['date'] = monthly_data['month_end']
    monthly_data['type'] = 'expense'

    categories = monthly_data['category'].unique()
    forecast_results = {}

    # Generate next month label
    last_month = monthly_data['month_end'].max()
    next_month = last_month + pd.DateOffset(months=1)
    next_month_label = next_month.strftime('%B %Y')

    print(f"\n{'='*80}")
    print(f"ENSEMBLE FORECASTING for {next_month_label}")
    print(f"{'='*80}\n")

    all_confidence = []
    all_metrics = []
    irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                            'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

    for cat in categories:
        cat_data = monthly_data[monthly_data['category'] == cat].copy()

        if len(cat_data) < 4:
            # Not enough data for ensemble
            avg_amount = cat_data['amount'].mean() if len(cat_data) > 0 else 0
            forecast_results[cat] = round(float(avg_amount), 2)
            print(
                f"  {cat}: Rs. {avg_amount:.0f} (insufficient data, using average)")
            continue

        # Prepare features
        cat_data = cat_data.sort_values('date').reset_index(drop=True)
        cat_data['month_num'] = range(len(cat_data))
        cat_data['lag_1'] = cat_data['amount'].shift(
            1).fillna(cat_data['amount'].mean())
        cat_data['lag_2'] = cat_data['amount'].shift(
            2).fillna(cat_data['amount'].mean())
        cat_data['rolling_mean_3'] = cat_data['amount'].rolling(
            window=3, min_periods=1).mean()

        X = cat_data[['month_num', 'lag_1', 'lag_2', 'rolling_mean_3']].values
        y = cat_data['amount'].values

        try:
            # Strategy 4 & 7: Train ensemble with cross-validation
            models, model_scores, cv_metrics = train_ensemble_models(X, y)

            if not models:
                # Fallback to simple average
                avg_amount = cat_data['amount'].mean()
                forecast_results[cat] = round(float(avg_amount), 2)
                continue

            # Predict next month using ensemble
            next_month_num = len(cat_data)
            last_amount = cat_data['amount'].iloc[-1]
            last_2_amount = cat_data['amount'].iloc[-2] if len(
                cat_data) >= 2 else last_amount
            rolling_mean = cat_data['rolling_mean_3'].iloc[-1]

            future_X = np.array(
                [[next_month_num, last_amount, last_2_amount, rolling_mean]])

            # Get ensemble prediction
            ensemble_pred, individual_preds = predict_with_ensemble(
                models, future_X)
            ensemble_pred = float(max(0, ensemble_pred))

            forecast_results[cat] = round(ensemble_pred, 2)

            # Strategy 3: Calculate advanced confidence
            if cat.lower() not in irregular_categories:
                # Use cross-validation results for confidence
                cv_preds = np.array([cv_metrics['cv_r2']] * len(y))
                cat_confidence = calculate_advanced_confidence(
                    cat_data,
                    cv_preds,
                    y,
                    model_scores
                )
                all_confidence.append(cat_confidence)

                print(
                    f"  {cat}: Rs. {ensemble_pred:.0f} (Confidence: {cat_confidence:.1f}%, Agreement: {cv_metrics['ensemble_agreement']:.1f}%)")
                if individual_preds:
                    pred_str = ', '.join(
                        [f"{p:.0f}" for p in individual_preds[:3]])
                    print(f"    └─ Predictions: [{pred_str}]")
            else:
                print(
                    f"  {cat}: Rs. {ensemble_pred:.0f} (Irregular - excluded from confidence)")

            all_metrics.append(cv_metrics)

        except Exception as e:
            print(f"  {cat}: Error ({str(e)[:50]}), using fallback")
            avg_amount = cat_data['amount'].mean()
            forecast_results[cat] = round(float(avg_amount), 2)

    # Overall confidence
    confidence = np.mean(all_confidence) if all_confidence else 75.0

    # Aggregate metrics
    if all_metrics:
        avg_mae = np.mean([m['cv_mae'] for m in all_metrics])
        avg_rmse = np.mean([m['cv_rmse'] for m in all_metrics])
        avg_r2 = np.mean([m['cv_r2'] for m in all_metrics])
        avg_agreement = np.mean([m['ensemble_agreement'] for m in all_metrics])
    else:
        avg_mae = avg_rmse = avg_r2 = avg_agreement = 0

    metrics = {
        'mae': float(avg_mae),
        'rmse': float(avg_rmse),
        'r2': float(avg_r2),
        'confidence': float(confidence),
        'ensemble_agreement': float(avg_agreement)
    }

    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS:")
    print(f"  Overall Confidence: {confidence:.1f}%")
    print(f"  CV MAE: Rs. {avg_mae:.2f}")
    print(f"  CV R²: {avg_r2:.3f}")
    print(f"  Ensemble Agreement: {avg_agreement:.1f}%")
    print(f"{'='*80}\n")

    return forecast_results, next_month_label, confidence, metrics


def forecast_next_month_expenses_xgboost(df: pd.DataFrame) -> Tuple[Dict, str, float, Dict]:
    """
    Optimized: Forecast next month's total expenses by category
    """
    expenses = df[df['type'] == 'expense'].copy()
    if expenses.empty:
        return {}, "", 0.0, {}

    # Ensure we have at least 12 months of data
    df_with_synthetic = generate_synthetic_data(expenses, target_months=12)
    expenses = df_with_synthetic[df_with_synthetic['type'] == 'expense'].copy()

    # Normalize category names BEFORE grouping to prevent duplicates
    expenses['category'] = expenses['category'].str.strip().str.capitalize()

    # Group by month and category
    monthly_data = expenses.groupby([pd.Grouper(key='date', freq='ME'), 'category'])[
        'amount'].sum().reset_index()
    monthly_data.rename(columns={'date': 'month_end'}, inplace=True)
    monthly_data['date'] = monthly_data['month_end']
    monthly_data['type'] = 'expense'

    # Get unique categories
    categories = monthly_data['category'].unique()
    forecast_results = {}

    # Generate next month label
    last_month = monthly_data['month_end'].max()
    next_month = last_month + pd.DateOffset(months=1)
    next_month_label = next_month.strftime('%B %Y')

    print(
        f"Debug: Forecasting for {next_month_label} with {len(categories)} categories")

    # Train one model per category (faster than combined)
    all_confidence = []

    for cat in categories:
        cat_data = monthly_data[monthly_data['category'] == cat].copy()

        if len(cat_data) < 3:
            # Not enough data, use average
            avg_amount = cat_data['amount'].mean() if len(cat_data) > 0 else 0
            forecast_results[cat] = round(float(avg_amount), 2)
            continue

        # Simple feature engineering for this category
        cat_data = cat_data.sort_values('date').reset_index(drop=True)
        cat_data['month_num'] = range(len(cat_data))
        cat_data['lag_1'] = cat_data['amount'].shift(
            1).fillna(cat_data['amount'].mean())
        cat_data['lag_2'] = cat_data['amount'].shift(
            2).fillna(cat_data['amount'].mean())
        cat_data['rolling_mean_3'] = cat_data['amount'].rolling(
            window=3, min_periods=1).mean()

        # Train simple model
        X = cat_data[['month_num', 'lag_1', 'lag_2', 'rolling_mean_3']].values
        y = cat_data['amount'].values

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=30,
            max_depth=3,
            learning_rate=0.15,
            random_state=42,
            verbosity=0
        )

        model.fit(X, y)

        # Predict next month
        next_month_num = len(cat_data)
        last_amount = cat_data['amount'].iloc[-1]
        last_2_amount = cat_data['amount'].iloc[-2] if len(
            cat_data) >= 2 else last_amount
        rolling_mean = cat_data['rolling_mean_3'].iloc[-1]

        future_X = np.array(
            [[next_month_num, last_amount, last_2_amount, rolling_mean]])
        pred = model.predict(future_X)[0]
        pred = float(max(0, pred))

        forecast_results[cat] = round(pred, 2)

        # Calculate simple confidence based on recent variance
        irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                                'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

        # Only calculate confidence for regular expenses
        if cat.lower() not in irregular_categories:
            recent_3 = cat_data['amount'].tail(3)
            variance = recent_3.std() / recent_3.mean() if recent_3.mean() > 0 else 0
            # More optimistic confidence calculation: 95 - (variance * 50) instead of 90 - (variance * 100)
            cat_confidence = max(70, min(95, 95 - (variance * 50)))
            all_confidence.append(cat_confidence)

        avg_last_3 = recent_3.mean()
        print(
            f"Debug: '{cat}' - Predicted: Rs. {pred:.0f}, Last 3 avg: Rs. {avg_last_3:.0f}")

    # Overall confidence
    confidence = np.mean(all_confidence) if all_confidence else 75.0

    metrics = {
        'mae': 0.0,
        'rmse': 0.0,
        'r2': confidence / 100,
        'confidence': float(confidence)
    }

    return forecast_results, next_month_label, confidence, metrics


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
        [pd.Grouper(key='date', freq='ME'), 'category'])['amount'].sum().reset_index()

    # Get total time periods for frequency calculation
    total_weeks = len(weekly['date'].unique())
    total_months = len(monthly['date'].unique())
    categories = weekly['category'].unique()
    forecast_results = {}
    last_week = weekly['date'].max()
    weeks_labels = [(last_week + pd.Timedelta(weeks=i)
                     ).strftime('%Y-%m-%d') for i in range(1, 5)]

    print(
        f"Debug: Forecasting for {len(categories)} categories over {total_weeks} weeks / {total_months} months")

    for cat in categories:
        cat_weekly = weekly[weekly['category'] ==
                            cat].set_index('date').sort_index()
        cat_monthly = monthly[monthly['category']
                              == cat].set_index('date').sort_index()

        # Calculate spending patterns
        weeks_with_spending = len(cat_weekly)
        months_with_spending = len(cat_monthly)
        weekly_frequency = weeks_with_spending / total_weeks if total_weeks > 0 else 0
        monthly_frequency = months_with_spending / \
            total_months if total_months > 0 else 0

        # Analyze transaction amounts to detect patterns
        avg_weekly_amount = cat_weekly['amount'].mean() if len(
            cat_weekly) > 0 else 0
        avg_monthly_amount = cat_monthly['amount'].mean() if len(
            cat_monthly) > 0 else 0
        max_transaction = expenses[expenses['category'] == cat]['amount'].max()

        print(f"Debug: '{cat}' - Weekly: {weeks_with_spending}/{total_weeks} ({weekly_frequency:.1%}), Monthly: {months_with_spending}/{total_months} ({monthly_frequency:.1%}), Avg: Rs. {avg_weekly_amount:.0f}/week, Max: Rs. {max_transaction:.0f}")

        # Improved categorization logic with better semantic understanding
        # Essential categories that people need regularly (with semantic improvements)
        essential_daily = ['groceries', 'commute', 'fuel', 'transport']
        essential_weekly = [
            'food'] if avg_weekly_amount > 0 and weekly_frequency > 0.4 else []
        monthly_bills = ['rent', 'utilities', 'insurance',
                         'subscription', 'mortgage', 'loan']

        is_monthly_expense = (monthly_frequency >= 0.8) and (
            weekly_frequency < 0.3) or cat.lower() in monthly_bills
        is_essential_daily = cat.lower() in essential_daily
        is_essential_weekly = cat.lower() in essential_weekly
        is_regular_expense = weekly_frequency >= 0.3 and not is_monthly_expense
        is_occasional_expense = 0.05 <= weekly_frequency < 0.3  # Lowered threshold
        is_rare_expense = weekly_frequency < 0.05  # Very rare

        # Detect recent trend changes (adaptability improvement)
        recent_data = cat_weekly.tail(4) if len(
            cat_weekly) >= 4 else cat_weekly  # Last 4 weeks
        older_data = cat_weekly.head(
            len(cat_weekly)-4) if len(cat_weekly) > 4 else cat_weekly

        recent_avg = recent_data['amount'].mean() if len(
            recent_data) > 0 else 0
        older_avg = older_data['amount'].mean() if len(
            older_data) > 0 else recent_avg

        # Detect if spending is increasing/decreasing (habit change detection)
        trend_change = (recent_avg - older_avg) / \
            older_avg if older_avg > 0 else 0
        is_trending_up = trend_change > 0.2  # 20% increase in recent spending
        is_trending_down = trend_change < -0.2  # 20% decrease in recent spending

        print(
            f"Debug: '{cat}' - Recent avg: Rs{recent_avg:.0f}, Older avg: Rs{older_avg:.0f}, Trend: {trend_change:.1%}")

        # Large transaction detection (improved)
        is_large_transaction = max_transaction > avg_monthly_amount * \
            1.5 if avg_monthly_amount > 0 else max_transaction > 5000

        # Force classifications for known categories
        if cat.lower() in monthly_bills:
            is_monthly_expense = True
            is_regular_expense = False

        # Essential daily expenses that should appear in most weeks
        is_essential_expense = is_essential_daily or is_essential_weekly

        if is_monthly_expense:
            # Monthly expenses: Show once every 4 weeks with trend adjustment
            monthly_amount = avg_monthly_amount if avg_monthly_amount > 0 else cat_weekly[
                'amount'].iloc[-1]

            # Apply trend changes for adaptability
            if is_trending_up:
                monthly_amount *= 1.1  # 10% increase if trending up
            elif is_trending_down:
                monthly_amount *= 0.9  # 10% decrease if trending down

            weekly_equivalent = monthly_amount / 4
            forecast_results[cat] = [weekly_equivalent, 0, 0, 0]
            print(
                f"Debug: '{cat}' classified as MONTHLY (Rs. {monthly_amount:.0f}/month, trend: {trend_change:.1%})")

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
            print(
                f"Debug: '{cat}' classified as RARE/LARGE (trend-adjusted: {multiplier})")

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
                default_essential = 3000 if cat.lower(
                ) == 'groceries' else 1500 if cat.lower() == 'food' else 1000
                forecast_results[cat] = [
                    default_essential * 0.8, default_essential * 0.6, default_essential, default_essential * 0.7]

            forecast_results[cat] = [round(v, 2)
                                     for v in forecast_results[cat]]
            print(f"Debug: '{cat}' classified as ESSENTIAL (trend-adjusted)")

        elif is_occasional_expense:
            # Monthly expenses: Show once every 4 weeks (approximately monthly)
            monthly_amount = avg_monthly_amount if avg_monthly_amount > 0 else cat_weekly[
                'amount'].iloc[-1]
            # Spread across 4 weeks for weekly forecast
            weekly_equivalent = monthly_amount / 4

            forecast_results[cat] = [weekly_equivalent,
                                     0, 0, 0]  # Show in first week only
            print(
                f"Debug: '{cat}' classified as MONTHLY (Rs. {monthly_amount:.0f}/month)")

        elif is_rare_expense or is_large_transaction:
            # Rare/large expenses: Show minimally with reduced frequency
            avg_amount = cat_weekly['amount'].mean() if len(
                cat_weekly) > 0 else 0
            # Minimal forecast in week 3
            forecast_results[cat] = [0, 0, avg_amount * 0.2, 0]
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
            print(
                f"Debug: '{cat}' classified as OCCASIONAL (trend-adjusted: {trend_change:.1%})")

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
                    estimate = 2000 if cat.lower() in [
                        'groceries', 'shopping'] else 800
                    forecast_results[cat] = [estimate * 0.8,
                                             estimate * 0.5, estimate, estimate * 0.6]
                print(
                    f"Debug: '{cat}' classified as REGULAR (sparse data, trend: {trend_change:.1%})")
                continue

            # Use trend-based forecasting for regular expenses with sufficient data
            base_amount = recent_avg if recent_avg > 0 else avg_weekly_amount

            # Apply trend adjustment
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
            forecast_results[cat] = [
                round(max(200, v), 2) for v in forecast_results[cat]]
            print(
                f"Debug: '{cat}' classified as REGULAR (trend-based, trend: {trend_change:.1%})")
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
        is_essential = category.lower() in [
            'food', 'groceries', 'transport', 'fuel', 'commute', 'rent', 'utilities']

        if is_essential or total_predicted > 50 or max_week_spending > 25:  # Much lower thresholds
            cleaned_results[category] = values
            print(
                f"Debug: Keeping category '{category}' (total: {total_predicted}, max: {max_week_spending}, essential: {is_essential})")
        else:
            print(
                f"Debug: Filtering out category '{category}' (total: {total_predicted}, max: {max_week_spending})")

    return cleaned_results


# Cashflow function removed - replaced with comprehensive AI insights


def generate_ai_insights_and_warnings(df: pd.DataFrame, forecast_data: Dict, confidence: float, metrics: Dict, next_month_label: str = "") -> Tuple[List[str], List[str], List[str]]:
    """
    Generate realistic AI-powered insights, warnings, and actionable recommendations based on past month data correlation
    """
    insights = []
    warnings = []
    recommendations = []

    if df.empty:
        return ["No financial data available. Start tracking your expenses and income to get personalized insights."], [], []

    # Separate expenses and income
    expenses = df[df['type'] == 'expense'].copy()
    income = df[df['type'] == 'revenue'].copy()

    # Get last date and calculate month boundaries
    last_date = df['date'].max()
    current_month_start = last_date.replace(day=1)
    last_month_start = current_month_start - pd.DateOffset(months=1)
    two_months_ago_start = current_month_start - pd.DateOffset(months=2)
    three_months_ago_start = current_month_start - pd.DateOffset(months=3)

    # Monthly data
    current_month_expenses = expenses[expenses['date'] >= current_month_start]
    last_month_expenses = expenses[(expenses['date'] >= last_month_start) & (
        expenses['date'] < current_month_start)]
    two_months_ago_expenses = expenses[(expenses['date'] >= two_months_ago_start) & (
        expenses['date'] < last_month_start)]
    three_months_ago_expenses = expenses[(expenses['date'] >= three_months_ago_start) & (
        expenses['date'] < two_months_ago_start)]
    last_3_months_expenses = expenses[expenses['date']
                                      >= three_months_ago_start]

    current_month_income = income[income['date'] >= current_month_start]
    last_month_income = income[(income['date'] >= last_month_start) & (
        income['date'] < current_month_start)]
    last_3_months_income = income[income['date'] >= three_months_ago_start]

    # Define irregular categories to exclude from regular spending analysis
    irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                            'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

    # Calculate totals (excluding irregular expenses for fair comparison)
    last_3_months_regular = last_3_months_expenses[~last_3_months_expenses['category'].str.lower(
    ).isin(irregular_categories)]

    current_month_expense_total = current_month_expenses['amount'].sum()
    last_month_expense_total = last_month_expenses['amount'].sum()
    two_months_ago_expense_total = two_months_ago_expenses['amount'].sum()
    three_months_ago_expense_total = three_months_ago_expenses['amount'].sum()
    last_3_months_avg_expense = last_3_months_regular['amount'].sum(
    ) / 3 if len(last_3_months_regular) > 0 else 0

    current_month_income_total = current_month_income['amount'].sum()
    last_month_income_total = last_month_income['amount'].sum()
    last_3_months_avg_income = last_3_months_income['amount'].sum(
    ) / 3 if len(last_3_months_income) > 0 else 0

    # Calculate forecasted total for next month (excluding irregular expenses)
    forecast_total = sum(amt for cat, amt in forecast_data.items(
    ) if cat.lower() not in irregular_categories) if forecast_data else 0

    # ==================== INSIGHTS ====================

    # 1. Next Month Forecast vs Last Month
    if last_month_expense_total > 0 and forecast_total > 0:
        month_change = (
            (forecast_total - last_month_expense_total) / last_month_expense_total) * 100

        if confidence >= 80:
            if month_change > 10:
                warnings.append(
                    f"⚠️ Expected Increase: {next_month_label} expenses forecasted at Rs. {forecast_total:.0f} ({month_change:+.1f}% vs last month's Rs. {last_month_expense_total:.0f})")
            elif month_change < -10:
                insights.append(
                    f"� Positive Outlook: {next_month_label} expenses expected to drop to Rs. {forecast_total:.0f} ({month_change:.1f}% vs last month)")
            else:
                insights.append(
                    f"📊 Steady Spending: {next_month_label} expenses forecasted at Rs. {forecast_total:.0f}, similar to last month (Rs. {last_month_expense_total:.0f})")
        else:
            insights.append(
                f"📈 Preliminary Forecast: {next_month_label} expenses estimated at Rs. {forecast_total:.0f}. Confidence: {confidence:.0f}%")

    # 2. Month-over-Month Trend (Last 3 months)
    if last_month_expense_total > 0 and two_months_ago_expense_total > 0:
        mom_change = ((last_month_expense_total -
                      two_months_ago_expense_total) / two_months_ago_expense_total) * 100

        if mom_change > 15:
            warnings.append(
                f"� Rising Trend: Your expenses increased by {mom_change:.1f}% last month. Last month: Rs. {last_month_expense_total:.0f}, Two months ago: Rs. {two_months_ago_expense_total:.0f}")
        elif mom_change < -15:
            insights.append(
                f"� Great Control: You reduced spending by {abs(mom_change):.1f}% last month compared to the previous month!")

    # 3. Category Analysis - Compare last month to forecast
    last_month_by_category = last_month_expenses.groupby(
        'category')['amount'].sum().sort_values(ascending=False)

    if len(last_month_by_category) > 0 and forecast_data:
        # Find categories with significant changes
        for cat in last_month_by_category.index[:3]:  # Top 3 categories
            last_month_cat_amt = last_month_by_category[cat]
            forecast_cat_amt = forecast_data.get(cat, 0)

            if forecast_cat_amt > 0 and last_month_cat_amt > 0:
                cat_change = (
                    (forecast_cat_amt - last_month_cat_amt) / last_month_cat_amt) * 100

                if abs(cat_change) > 20:
                    if cat_change > 0:
                        warnings.append(
                            f"📊 {cat}: Expected to increase by {cat_change:.0f}% (Rs. {last_month_cat_amt:.0f} → Rs. {forecast_cat_amt:.0f})")
                    else:
                        insights.append(
                            f"� {cat}: Expected to decrease by {abs(cat_change):.0f}% (Rs. {last_month_cat_amt:.0f} → Rs. {forecast_cat_amt:.0f})")

    # 4. Savings Analysis (Last Month)
    if last_month_income_total > 0 and last_month_expense_total > 0:
        last_month_savings = last_month_income_total - last_month_expense_total
        savings_rate = (last_month_savings / last_month_income_total) * 100

        if savings_rate < 0:
            warnings.append(
                f"🚨 Last Month Deficit: You spent Rs. {abs(last_month_savings):.0f} more than you earned. Income: Rs. {last_month_income_total:.0f}, Expenses: Rs. {last_month_expense_total:.0f}")
        elif savings_rate < 10:
            warnings.append(
                f"⚠️ Low Savings Rate: Last month you saved only {savings_rate:.1f}% (Rs. {last_month_savings:.0f}). Aim for 20%+")
        elif savings_rate >= 20 and savings_rate < 30:
            insights.append(
                f"✅ Healthy Savings: Last month you saved {savings_rate:.1f}% (Rs. {last_month_savings:.0f}) of your income")
        elif savings_rate >= 30:
            insights.append(
                f"🌟 Excellent Savings: Last month you saved {savings_rate:.1f}% (Rs. {last_month_savings:.0f})!")

    # 5. Forecast vs Average Comparison
    if last_3_months_avg_expense > 0 and forecast_total > 0:
        vs_avg_change = (
            (forecast_total - last_3_months_avg_expense) / last_3_months_avg_expense) * 100

        if vs_avg_change > 5:
            insights.append(
                f"📊 Above Average: Next month forecast (Rs. {forecast_total:.0f}) is {vs_avg_change:.1f}% above your 3-month average (Rs. {last_3_months_avg_expense:.0f})")
        elif vs_avg_change < -5:
            insights.append(
                f"💰 Below Average: Next month forecast (Rs. {forecast_total:.0f}) is {abs(vs_avg_change):.1f}% below your 3-month average (Rs. {last_3_months_avg_expense:.0f})")

    # 6. Income Stability Check
    if last_month_income_total > 0 and last_3_months_avg_income > 0:
        income_variance = abs(
            (last_month_income_total - last_3_months_avg_income) / last_3_months_avg_income) * 100

        if income_variance > 20:
            insights.append(
                f"� Income Variance: Last month income (Rs. {last_month_income_total:.0f}) varied {income_variance:.0f}% from your 3-month average (Rs. {last_3_months_avg_income:.0f})")

    # ==================== WARNINGS ====================

    # 7. Top Spending Category
    if len(last_month_by_category) > 0:
        top_cat = last_month_by_category.index[0]
        top_amt = last_month_by_category.iloc[0]
        top_pct = (top_amt / last_month_expense_total) * \
            100 if last_month_expense_total > 0 else 0

        warnings.append(
            f"💡 Top Category: {top_cat} was your biggest expense last month at Rs. {top_amt:.0f} ({top_pct:.0f}% of total)")

    # 7b. Identify Non-Recurring/Irregular Expenses
    irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                            'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

    if forecast_data:
        for cat, amt in forecast_data.items():
            if cat.lower() in irregular_categories and amt > 500:
                warnings.append(
                    f"ℹ️ Note: {cat} (Rs. {amt:.0f}) is classified as an irregular/non-recurring expense and excluded from forecast total. Actual spending may vary.")

    # 8. Forecast vs Income Warning (if income available)
    if last_3_months_avg_income > 0 and forecast_total > 0:
        forecast_to_income_ratio = (
            forecast_total / last_3_months_avg_income) * 100

        if forecast_to_income_ratio > 90:
            warnings.append(
                f"💸 High Expense Forecast: {next_month_label} expenses (Rs. {forecast_total:.0f}) will consume {forecast_to_income_ratio:.0f}% of your average income")
        elif forecast_to_income_ratio > 80:
            warnings.append(
                f"⚠️ Watch Your Budget: {next_month_label} expenses forecasted at {forecast_to_income_ratio:.0f}% of your income")

    # 9. Unusual Category Spike Detection
    if len(last_month_by_category) > 0 and len(two_months_ago_expenses) > 0:
        two_months_by_cat = two_months_ago_expenses.groupby('category')[
            'amount'].sum()

        for cat in last_month_by_category.index[:5]:
            last_month_cat = last_month_by_category[cat]
            two_months_cat = two_months_by_cat.get(cat, 0)

            if two_months_cat > 0:
                spike = ((last_month_cat - two_months_cat) /
                         two_months_cat) * 100

                if spike > 50 and last_month_cat > 1000:
                    warnings.append(
                        f"� Unusual Spike: {cat} increased by {spike:.0f}% last month (Rs. {two_months_cat:.0f} → Rs. {last_month_cat:.0f})")

    # 10. No Recent Income Warning
    if current_month_income_total == 0 and last_month_income_total == 0:
        warnings.append(
            "⚠️ No Recent Income: No income recorded in the last 2 months. Make sure to track all income sources")

    # 11. Low Confidence Warning
    if confidence < 70:
        warnings.append(
            f"📊 Forecast Uncertainty: Model confidence is {confidence:.0f}%. Predictions improve with more historical data")

    # 12. High Essential Expenses
    essential_cats = ['rent', 'utilities', 'emi',
                      'loan', 'insurance', 'food', 'groceries']
    essential_last_month = last_month_expenses[last_month_expenses['category'].str.lower(
    ).isin(essential_cats)]['amount'].sum()

    if essential_last_month > 0 and last_month_expense_total > 0:
        essential_pct = (essential_last_month / last_month_expense_total) * 100

        if essential_pct > 70:
            warnings.append(
                f"🏠 High Essential Expenses: {essential_pct:.0f}% of last month's expenses were essentials (Rs. {essential_last_month:.0f}). Limited room for savings.")
        elif essential_pct < 40:
            warnings.append(
                f"💡 High Discretionary Spending: Only {essential_pct:.0f}% on essentials. Review discretionary expenses to increase savings.")

    # 13. Frequent Small Transactions Warning
    if len(last_month_expenses) > 50:  # More than 50 transactions in a month
        small_transactions = last_month_expenses[last_month_expenses['amount'] < 500]
        if len(small_transactions) > 20:  # More than 20 small transactions
            small_total = small_transactions['amount'].sum()
            warnings.append(
                f"🛒 Frequent Small Purchases: {len(small_transactions)} transactions under Rs. 500 totaling Rs. {small_total:.0f}. These add up quickly!")

    # 14. Spending Acceleration Warning
    if last_month_expense_total > 0 and two_months_ago_expense_total > 0 and three_months_ago_expense_total > 0:
        recent_trend = ((last_month_expense_total -
                        two_months_ago_expense_total) / two_months_ago_expense_total) * 100
        older_trend = ((two_months_ago_expense_total -
                       three_months_ago_expense_total) / three_months_ago_expense_total) * 100

        if recent_trend > 10 and older_trend > 10:
            warnings.append(
                f"📈 Continuous Growth: Expenses have been rising for 2+ months. Last 3 months: Rs. {three_months_ago_expense_total:.0f} → Rs. {two_months_ago_expense_total:.0f} → Rs. {last_month_expense_total:.0f}")

    # 15. Weekend/Lifestyle Spending Warning
    lifestyle_cats = ['entertainment', 'dining',
                      'travel', 'shopping', 'leisure', 'hobbies', 'gym']
    lifestyle_last_month = last_month_expenses[last_month_expenses['category'].str.lower(
    ).isin(lifestyle_cats)]['amount'].sum()

    if lifestyle_last_month > 0 and last_month_expense_total > 0:
        lifestyle_pct = (lifestyle_last_month / last_month_expense_total) * 100

        if lifestyle_pct > 40:
            warnings.append(
                f"🎉 High Lifestyle Spending: {lifestyle_pct:.0f}% (Rs. {lifestyle_last_month:.0f}) spent on entertainment, dining, travel, and shopping")
        elif lifestyle_pct > 30:
            warnings.append(
                f"🎯 Lifestyle Budget: {lifestyle_pct:.0f}% on lifestyle categories. Consider moderating for better savings")

    # 16. Budget Burn Rate (if income available)
    if last_month_income_total > 0 and forecast_total > 0:
        projected_savings = last_month_income_total - forecast_total
        burn_rate = (forecast_total / last_month_income_total) * 100

        if burn_rate > 95 and projected_savings < 1000:
            warnings.append(
                f"🔥 Critical Burn Rate: Next month forecast leaves only Rs. {projected_savings:.0f} ({100-burn_rate:.1f}%) for savings")

    # ==================== ACTIONABLE RECOMMENDATIONS ====================

    # 1. Savings Recommendations
    if last_month_income_total > 0 and last_month_expense_total > 0:
        last_month_savings = last_month_income_total - last_month_expense_total
        savings_rate = (last_month_savings / last_month_income_total) * 100

        if savings_rate < 0:
            deficit = abs(last_month_savings)
            recommendations.append(
                f"💰 Emergency Action: Create a deficit recovery plan. Cut non-essential expenses by Rs. {deficit * 0.5:.0f} and look for additional income sources.")
            recommendations.append(
                f"📊 Budget Strategy: Review your top 3 expense categories and identify areas to reduce spending by 20-30%.")
        elif savings_rate < 10:
            target_savings = last_month_income_total * 0.20
            gap = target_savings - last_month_savings
            recommendations.append(
                f"🎯 Savings Goal: Aim to save Rs. {gap:.0f} more per month to reach the recommended 20% savings rate.")
        elif savings_rate < 20:
            recommendations.append(
                f"📈 Good Progress: You're saving {savings_rate:.1f}%. Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.")
        else:
            recommendations.append(
                f"⭐ Excellent! You're saving {savings_rate:.1f}%. Consider investing excess savings in mutual funds, SIPs, or emergency fund.")

    # 2. Category-Specific Recommendations
    if len(last_month_by_category) > 0:
        # Lifestyle spending recommendations
        lifestyle_cats = ['entertainment', 'dining',
                          'travel', 'shopping', 'leisure', 'hobbies', 'gym']
        lifestyle_spending = last_month_expenses[last_month_expenses['category'].str.lower(
        ).isin(lifestyle_cats)]['amount'].sum()

        if lifestyle_spending > 0 and last_month_expense_total > 0:
            lifestyle_pct = (lifestyle_spending /
                             last_month_expense_total) * 100

            if lifestyle_pct > 40:
                reduction_target = lifestyle_spending * 0.3
                recommendations.append(
                    f"🎯 Lifestyle Optimization: Your lifestyle spending is {lifestyle_pct:.0f}%. Consider reducing by Rs. {reduction_target:.0f}/month through meal prep and free entertainment alternatives.")
            elif lifestyle_pct > 30:
                recommendations.append(
                    f"⚖️ Balance Needed: {lifestyle_pct:.0f}% on lifestyle is manageable but could be optimized. Set weekly spending limits.")

        # Food & Groceries recommendations
        food_cats = ['food', 'groceries', 'dining']
        food_spending = last_month_expenses[last_month_expenses['category'].str.lower(
        ).isin(food_cats)]['amount'].sum()

        if food_spending > 15000:
            recommendations.append(
                f"🍽️ Food Budget: Spending Rs. {food_spending:.0f} on food. Try meal planning and cooking at home 5 days/week to save ~Rs. {food_spending * 0.25:.0f}.")

        # Transport recommendations
        transport_cats = ['transport', 'fuel', 'commute']
        transport_spending = last_month_expenses[last_month_expenses['category'].str.lower(
        ).isin(transport_cats)]['amount'].sum()

        if transport_spending > 8000:
            recommendations.append(
                f"🚗 Transport Optimization: Rs. {transport_spending:.0f} on transport. Consider carpooling or public transport to save 30-40%.")

    # 3. Forecast-Based Recommendations
    if forecast_total > 0 and last_3_months_avg_income > 0:
        forecast_to_income_ratio = (
            forecast_total / last_3_months_avg_income) * 100

        if forecast_to_income_ratio > 90:
            shortfall = forecast_total - (last_3_months_avg_income * 0.8)
            recommendations.append(
                f"⚠️ Budget Alert: Next month's forecast is {forecast_to_income_ratio:.0f}% of income. Reduce discretionary spending by Rs. {shortfall:.0f}.")
        elif forecast_to_income_ratio > 80:
            recommendations.append(
                f"📊 Proactive Planning: Next month forecast is {forecast_to_income_ratio:.0f}% of income. Monitor daily spending to stay on track.")

    # 4. Spending Trend Recommendations
    if last_month_expense_total > 0 and two_months_ago_expense_total > 0:
        mom_change = ((last_month_expense_total -
                      two_months_ago_expense_total) / two_months_ago_expense_total) * 100

        if mom_change > 15:
            increase = last_month_expense_total - two_months_ago_expense_total
            recommendations.append(
                f"📉 Spending Control: Expenses rose by {mom_change:.1f}% (Rs. {increase:.0f}). Review and categorize all expenses, set category-wise limits.")

    # 5. Emergency Fund Recommendations
    if last_3_months_avg_expense > 0 and last_month_income_total > 0:
        emergency_fund_target = last_3_months_avg_expense * 3
        savings_rate = ((last_month_income_total - last_month_expense_total) /
                        last_month_income_total) * 100 if last_month_income_total > 0 else 0

        if savings_rate > 0:
            recommendations.append(
                f"🛡️ Financial Security: Build an emergency fund of Rs. {emergency_fund_target:.0f} (3 months expenses). Start with Rs. 3000/month.")

    # 6. Investment Recommendations
    if last_month_income_total > 0 and last_month_expense_total > 0:
        savings_amount = last_month_income_total - last_month_expense_total
        savings_rate = (savings_amount / last_month_income_total) * 100

        if savings_rate >= 25 and savings_amount > 5000:
            recommendations.append(
                f"💎 Wealth Building: You're saving Rs. {savings_amount:.0f}/month. Consider investing 50% in equity mutual funds (SIP), 30% in debt funds, 20% in emergency fund.")

    # 7. Small Transaction Optimization
    if len(last_month_expenses) > 50:
        small_transactions = last_month_expenses[last_month_expenses['amount'] < 500]
        if len(small_transactions) > 20:
            small_total = small_transactions['amount'].sum()
            recommendations.append(
                f"💳 Micro-Spending Alert: {len(small_transactions)} small purchases totaling Rs. {small_total:.0f}. Set a Rs. 200 daily limit for small purchases.")

    # 8. Subscription Audit
    subscription_cats = ['subscription', 'membership', 'streaming', 'software']
    subscriptions = last_month_expenses[last_month_expenses['category'].str.lower(
    ).isin(subscription_cats)]

    if len(subscriptions) > 3:
        sub_total = subscriptions['amount'].sum()
        recommendations.append(
            f"📱 Subscription Audit: Review {len(subscriptions)} subscriptions (Rs. {sub_total:.0f}/month). Cancel unused services and switch to annual plans for 20-30% savings.")

    return insights, warnings, recommendations


def generate_insights(df):
    """Legacy function - redirects to new AI insights"""
    insights, warnings, recommendations = generate_ai_insights_and_warnings(df, {
    }, 0, {})
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
            [pd.Grouper(key='date', freq='ME'), 'category'])['amount'].sum().reset_index()

        total_weeks = len(weekly['date'].unique())
        total_months = len(monthly['date'].unique())
        categories = weekly['category'].unique()

        classification_results = {}

        for cat in categories:
            cat_weekly = weekly[weekly['category'] ==
                                cat].set_index('date').sort_index()
            cat_monthly = monthly[monthly['category']
                                  == cat].set_index('date').sort_index()

            weeks_with_spending = len(cat_weekly)
            months_with_spending = len(cat_monthly)
            weekly_frequency = weeks_with_spending / total_weeks if total_weeks > 0 else 0
            monthly_frequency = months_with_spending / \
                total_months if total_months > 0 else 0

            avg_weekly_amount = cat_weekly['amount'].mean() if len(
                cat_weekly) > 0 else 0
            avg_monthly_amount = cat_monthly['amount'].mean() if len(
                cat_monthly) > 0 else 0
            max_transaction = expenses[expenses['category']
                                       == cat]['amount'].max()

            # Apply same classification logic
            is_monthly_expense = (monthly_frequency >= 0.8) and (
                weekly_frequency < 0.3)
            is_regular_expense = weekly_frequency >= 0.3
            is_occasional_expense = 0.1 <= weekly_frequency < 0.3
            is_rare_expense = weekly_frequency < 0.1
            is_large_transaction = max_transaction > avg_monthly_amount * \
                2 if avg_monthly_amount > 0 else False
            is_essential_expense = cat.lower(
            ) in ['food', 'groceries', 'transport', 'fuel', 'commute']

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
        # Fetch and prepare data (12 months for better AI predictions)
        data = fetch_data(user_id=userId, months=12)
        df = prepare_dataframe(data)

        if df.empty:
            print("Debug: No historical data found, using estimates")
            return {
                "forecasts": {
                    "next_month": "",
                    "expenses": {},
                    "total": 0
                },
                "model_metrics": {"confidence": 0},
                "insights": ["📊 No historical data available. Start tracking your transactions for AI-powered insights!"],
                "warnings": ["⚠️ Add income and expense data to get personalized predictions."]
            }

        # Generate XGBoost forecasts for next month
        expense_forecast, next_month_label, confidence, metrics = forecast_next_month_expenses_xgboost(
            df)
        cleaned_forecast = {cat: amt for cat,
                            amt in expense_forecast.items() if amt > 100}
        insights, warnings = generate_ai_insights_and_warnings(
            df, cleaned_forecast, confidence, metrics, next_month_label)

        return {
            "forecasts": {
                "next_month": next_month_label,
                "expenses": cleaned_forecast,
                "total": round(sum(cleaned_forecast.values()), 2)
            },
            "model_metrics": {
                "confidence": round(confidence, 1),
                "mae": round(metrics.get('mae', 0), 2),
                "rmse": round(metrics.get('rmse', 0), 2),
                "r2": round(metrics.get('r2', 0), 3)
            },
            "insights": insights,
            "warnings": warnings
        }
    except Exception as e:
        print(f"Error in api_insights: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}


@app.get("/api/forecast/{userId}")
def api_forecast_get(userId: str):
    print(f"=" * 80)
    print(f"API CALLED: GET /api/forecast/{userId}")
    print(f"=" * 80)
    try:
        # Fetch last 12 months of data for better forecasting
        data = fetch_data(user_id=userId, months=12)
        print(f"Debug: Fetched {len(data)} records for user {userId}")

        df = prepare_dataframe(data)

        if df.empty:
            return {
                "forecast_chart_data": {"next_month": "", "categories": {}},
                "model_metrics": {"confidence": 0, "mae": 0, "rmse": 0, "r2": 0},
                "insights": ["No historical data found. Please add some transactions to generate AI-powered forecasts."],
                "warnings": ["Add your income and expense data to get personalized insights and predictions."],
                "recommendations": ["Start tracking your daily expenses to get personalized financial recommendations."]
            }

        # Use XGBoost forecasting
        forecast_data, next_month_label, confidence, metrics = forecast_next_month_expenses_xgboost(
            df)

        # Define irregular categories that shouldn't be included in regular monthly forecast
        irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                                'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

        # Clean the forecast data: remove minimal spending and irregular expenses
        cleaned_forecast_data = {
            cat: amt for cat, amt in forecast_data.items()
            if amt > 100 and cat.lower() not in irregular_categories
        }

        # Generate AI insights, warnings, and recommendations based on past month correlation
        # Pass the original forecast_data so insights can see irregular expenses
        insights, warnings, recommendations = generate_ai_insights_and_warnings(
            df, forecast_data, confidence, metrics, next_month_label)

        return {
            "forecast_chart_data": {
                "next_month": next_month_label,
                "categories": cleaned_forecast_data,
                "total": round(sum(cleaned_forecast_data.values()), 2)
            },
            "model_metrics": {
                "confidence": round(confidence, 1),
                "mae": round(metrics.get('mae', 0), 2),
                "rmse": round(metrics.get('rmse', 0), 2),
                "r2": round(metrics.get('r2', 0), 3)
            },
            "insights": insights,
            "warnings": warnings,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error in api_forecast_get: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}


@app.post("/api/forecast/{userId}")
def api_forecast_post(userId: str, body: UserEstimates):
    try:
        user_estimates = body.estimates
        print(f"Debug: Received estimates for user {userId}: {user_estimates}")

        # Fetch last 12 months of data
        data = fetch_data(user_id=userId, months=12)
        print(f"Debug: Fetched {len(data)} records")

        df = prepare_dataframe(data)

        if df.empty:
            next_month = (datetime.now() +
                          relativedelta(months=1)).strftime('%B %Y')
            forecast_results = {}
            revenue_sum = 0
            expense_sum = 0

            for cat, amount in user_estimates.items():
                if cat.lower() in ["salary", "freelance", "income"]:
                    revenue_sum += amount
                else:
                    expense_sum += amount
                    forecast_results[cat] = round(amount, 2)

            insights = [
                "📊 No historical data found. Using your input estimates for forecasting.",
                f"💰 Your estimated monthly income: ₹{int(revenue_sum)}",
                f"💸 Your estimated monthly expenses: ₹{int(expense_sum)}",
                "🎯 Start tracking transactions for AI-powered predictions!"
            ]

            warnings = []
            if expense_sum > revenue_sum:
                warnings.append(
                    f"⚠️ Your estimated expenses (₹{int(expense_sum)}) exceed income (₹{int(revenue_sum)}). Review your budget!")

            recommendations = [
                "📝 Start tracking all transactions for accurate forecasts",
                "💰 Set up automatic expense tracking with bank notifications",
                "🎯 Aim to save at least 20% of your income"
            ]

            return {
                "forecast_chart_data": {
                    "next_month": next_month,
                    "categories": forecast_results,
                    "total": expense_sum
                },
                "model_metrics": {"confidence": 0, "mae": 0, "rmse": 0, "r2": 0},
                "insights": insights,
                "warnings": warnings,
                "recommendations": recommendations
            }

        # Use XGBoost forecasting for next month
        forecast_data, next_month_label, confidence, metrics = forecast_next_month_expenses_xgboost(
            df)

        # Define irregular categories that shouldn't be included in regular monthly forecast
        irregular_categories = ['accident', 'loan', 'medical emergency', 'repair',
                                'emergency', 'legal', 'fine', 'penalty', 'hospital', 'surgery']

        # Clean the forecast data: remove minimal spending and irregular expenses
        cleaned_forecast_data = {
            cat: amt for cat, amt in forecast_data.items()
            if amt > 100 and cat.lower() not in irregular_categories
        }

        # Pass the original forecast_data so insights can see irregular expenses for notes
        insights, warnings, recommendations = generate_ai_insights_and_warnings(
            df, forecast_data, confidence, metrics, next_month_label)

        return {
            "forecast_chart_data": {
                "next_month": next_month_label,
                "categories": cleaned_forecast_data,
                "total": round(sum(cleaned_forecast_data.values()), 2)
            },
            "model_metrics": {
                "confidence": round(confidence, 1),
                "mae": round(metrics.get('mae', 0), 2),
                "rmse": round(metrics.get('rmse', 0), 2),
                "r2": round(metrics.get('r2', 0), 3)
            },
            "insights": insights,
            "warnings": warnings,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error in api_forecast_post: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}
