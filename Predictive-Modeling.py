# predictive_modeling.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
demand_forecast_model_path = 'models/demand_forecast_model.pkl'
lead_time_prediction_model_path = 'models/lead_time_prediction_model.pkl'
results_path = 'results/predictive_modeling_report.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(demand_forecast_model_path), exist_ok=True)
os.makedirs(os.path.dirname(lead_time_prediction_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Prepare data for demand forecasting
X_demand = data.drop(columns=['demand'])
y_demand = data['demand']

# Split the data into training and testing sets for demand forecasting
X_demand_train, X_demand_test, y_demand_train, y_demand_test = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)

# Prepare data for lead time prediction
X_lead_time = data.drop(columns=['lead_time'])
y_lead_time = data['lead_time']

# Split the data into training and testing sets for lead time prediction
X_lead_time_train, X_lead_time_test, y_lead_time_train, y_lead_time_test = train_test_split(X_lead_time, y_lead_time, test_size=0.2, random_state=42)

# Function to evaluate and save model
def evaluate_and_save_model(model, X_train, y_train, X_test, y_test, model_name, model_path):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    with open(results_path, 'a') as f:
        f.write(f"{model_name} Performance:\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write("\n")
    
    print(f"Saving {model_name}...")
    joblib.dump(model, model_path)

# Train and evaluate demand forecasting models
linear_regression_model = LinearRegression()
random_forest_model = RandomForestRegressor(random_state=42)

evaluate_and_save_model(linear_regression_model, X_demand_train, y_demand_train, X_demand_test, y_demand_test, 'Linear Regression (Demand Forecast)', demand_forecast_model_path)
evaluate_and_save_model(random_forest_model, X_demand_train, y_demand_train, X_demand_test, y_demand_test, 'Random Forest (Demand Forecast)', demand_forecast_model_path)

# Train and evaluate lead time prediction models
evaluate_and_save_model(linear_regression_model, X_lead_time_train, y_lead_time_train, X_lead_time_test, y_lead_time_test, 'Linear Regression (Lead Time Prediction)', lead_time_prediction_model_path)
evaluate_and_save_model(random_forest_model, X_lead_time_train, y_lead_time_train, X_lead_time_test, y_lead_time_test, 'Random Forest (Lead Time Prediction)', lead_time_prediction_model_path)

print("Predictive modeling completed!")
