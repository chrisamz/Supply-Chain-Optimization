# exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Basic statistics
print("Basic Statistics:")
print(data.describe())

# Correlation matrix
print("Correlation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(figures_path, 'correlation_matrix.png'))
plt.show()

# Distribution of numerical features
numerical_features = ['cost_of_goods_sold', 'average_inventory', 'order_lead_time', 'number_of_orders', 'inventory_turnover', 'order_cycle_time']
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_path, f'{feature}_distribution.png'))
    plt.show()

# Box plots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature])
    plt.title(f'{feature.capitalize()} Box Plot')
    plt.xlabel(feature.capitalize())
    plt.savefig(os.path.join(figures_path, f'{feature}_boxplot.png'))
    plt.show()

# Categorical feature distribution
categorical_features = data.select_dtypes(include=['int64']).columns
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, data=data)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.savefig(os.path.join(figures_path, f'{feature}_countplot.png'))
    plt.show()

print("Exploratory Data Analysis completed!")
