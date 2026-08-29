import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

data = pd.read_csv(PROCESSED_DIR / "training_data.csv")

CATEGORICAL_FEATURES = ['batting_team', 'bowling_team', 'venue']
NUMERIC_FEATURES = ['over', 'cumulative_score', 'cumulative_wickets',
                    'last_5_overs_runs', 'last_5_overs_wickets']
TARGET = 'final_score'

# One-hot encode the text columns
data_encoded = pd.get_dummies(data, columns=CATEGORICAL_FEATURES)

# The feature columns are now: original numeric ones + all the new one-hot columns
feature_cols = NUMERIC_FEATURES + \
    [c for c in data_encoded.columns if c.startswith(
        tuple(f"{cat}_" for cat in CATEGORICAL_FEATURES))]

X = data_encoded[feature_cols]
y = data_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Total features:", len(feature_cols))
print("Training rows:", len(X_train))


# Baseline: project current run rate forward to 20 overs
# over is zero-indexed, so add 1 to get the actual number of overs completed
overs_completed = X_test['over'] + 1
current_run_rate = X_test['cumulative_score'] / overs_completed
baseline_predictions = current_run_rate * 20

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

baseline_mape = np.mean(np.abs(
    (y_test - baseline_predictions) / baseline_predictions.replace(0, np.nan))) * 100
print(
    f"Baseline  -> MAE: {baseline_mae:.2f}  RMSE: {baseline_rmse:.2f}  R2: {baseline_r2:.3f}  Avg Error: {baseline_mape:.1f}%")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    results[name] = model
    print(f"{name:20s} -> MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.3f}  Avg Error: {mape:.1f}%")


MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

best_model = results["XGBoost"]
joblib.dump(best_model, MODELS_DIR / "xgb_model.pkl")

# We also need to save the exact list of feature columns used,
# since the app will need to build inputs in this exact same order/format later
joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")

print("\nModel and feature columns saved to models/")
