# optimization_modeling.py

import pandas as pd
from scipy.optimize import linprog
import numpy as np
import os

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
results_path = 'results/optimization_results.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Extract relevant columns for optimization
cost_of_goods_sold = data['cost_of_goods_sold'].values
average_inventory = data['average_inventory'].values
order_lead_time = data['order_lead_time'].values
number_of_orders = data['number_of_orders'].values

# Define the objective function coefficients
# Objective: Minimize the total cost which is a combination of holding costs and ordering costs
holding_cost_per_unit = 0.1  # Example holding cost per unit
ordering_cost_per_order = 50  # Example ordering cost per order

c = holding_cost_per_unit * average_inventory + ordering_cost_per_order * number_of_orders

# Define the inequality constraints (Ax <= b)
# Example constraint: Demand must be met
demand = data['demand'].values
A = np.zeros((len(demand), len(demand)))
np.fill_diagonal(A, 1)
b = demand

# Define the bounds for each decision variable
# Example: Inventory levels and order quantities must be non-negative
x_bounds = [(0, None) for _ in range(len(demand))]

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
    f.write("Optimization Results:\n")
    f.write(f"Optimized Inventory Levels: {optimized_inventory_levels}\n")
    f.write(f"Optimized Total Cost: {optimized_total_cost}\n")

print("Optimization results saved!")
