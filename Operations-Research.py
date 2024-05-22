# operations_research.py

import pandas as pd
import joblib
from scipy.optimize import linprog
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
demand_forecast_model_path = 'models/demand_forecast_model.pkl'
lead_time_prediction_model_path = 'models/lead_time_prediction_model.pkl'
results_path = 'results/operations_research_results.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Load trained models
print("Loading trained models...")
demand_forecast_model = joblib.load(demand_forecast_model_path)
lead_time_prediction_model = joblib.load(lead_time_prediction_model_path)

# Prepare data for predictions
X = data.drop(columns=['demand', 'lead_time'])

# Predict future demand
print("Predicting future demand...")
predicted_demand = demand_forecast_model.predict(X)
data['predicted_demand'] = predicted_demand

# Predict lead time
print("Predicting lead time...")
predicted_lead_time = lead_time_prediction_model.predict(X)
data['predicted_lead_time'] = predicted_lead_time

# Evaluate predictions
print("Evaluating predictions...")
demand_mae = mean_absolute_error(data['demand'], data['predicted_demand'])
demand_rmse = np.sqrt(mean_squared_error(data['demand'], data['predicted_demand']))
demand_r2 = r2_score(data['demand'], data['predicted_demand'])

lead_time_mae = mean_absolute_error(data['lead_time'], data['predicted_lead_time'])
lead_time_rmse = np.sqrt(mean_squared_error(data['lead_time'], data['predicted_lead_time']))
lead_time_r2 = r2_score(data['lead_time'], data['predicted_lead_time'])

print(f"Demand Forecast MAE: {demand_mae:.4f}")
print(f"Demand Forecast RMSE: {demand_rmse:.4f}")
print(f"Demand Forecast R2: {demand_r2:.4f}")

print(f"Lead Time Prediction MAE: {lead_time_mae:.4f}")
print(f"Lead Time Prediction RMSE: {lead_time_rmse:.4f}")
print(f"Lead Time Prediction R2: {lead_time_r2:.4f}")

# Optimization
print("Starting optimization...")

# Define the objective function coefficients
holding_cost_per_unit = 0.1  # Example holding cost per unit
ordering_cost_per_order = 50  # Example ordering cost per order
c = holding_cost_per_unit * data['average_inventory'] + ordering_cost_per_order * data['number_of_orders']

# Define the inequality constraints (Ax <= b)
A = np.zeros((len(predicted_demand), len(predicted_demand)))
np.fill_diagonal(A, 1)
b = predicted_demand

# Define the bounds for each decision variable
x_bounds = [(0, None) for _ in range(len(predicted_demand))]

# Solve the linear programming problem
print("Solving the optimization problem...")
res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

# Extract the optimized values
optimized_inventory_levels = res.x
optimized_total_cost = res.fun

print("Optimization completed!")
print(f"Optimized Inventory Levels: {optimized_inventory_levels}")
print(f"Optimized Total Cost: {optimized_total_cost}")

# Save the optimization results
with open(results_path, 'w') as f:
    f.write("Operations Research and Machine Learning Results:\n")
    f.write(f"Demand Forecast MAE: {demand_mae:.4f}\n")
    f.write(f"Demand Forecast RMSE: {demand_rmse:.4f}\n")
    f.write(f"Demand Forecast R2: {demand_r2:.4f}\n")
    f.write(f"Lead Time Prediction MAE: {lead_time_mae:.4f}\n")
    f.write(f"Lead Time Prediction RMSE: {lead_time_rmse:.4f}\n")
    f.write(f"Lead Time Prediction R2: {lead_time_r2:.4f}\n")
    f.write(f"Optimized Inventory Levels: {optimized_inventory_levels}\n")
    f.write(f"Optimized Total Cost: {optimized_total_cost}\n")

print("Operations research and machine learning results saved!")
