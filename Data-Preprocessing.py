# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Define file paths
raw_data_path = 'data/raw/supply_chain_data.csv'
processed_data_path = 'data/processed/processed_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Data Cleaning
print("Cleaning data...")

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Feature Engineering
print("Performing feature engineering...")

# Example: Create new features based on existing data
data['inventory_turnover'] = data['cost_of_goods_sold'] / data['average_inventory']
data['order_cycle_time'] = data['order_lead_time'] / data['number_of_orders']

# Normalize numerical features
print("Normalizing numerical features...")
scaler = StandardScaler()
numerical_features = ['cost_of_goods_sold', 'average_inventory', 'order_lead_time', 'number_of_orders', 'inventory_turnover', 'order_cycle_time']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save processed data
print("Saving processed data...")
data.to_csv(processed_data_path, index=False)

print("Data preprocessing completed!")
