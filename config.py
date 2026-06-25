"""
Configuration file for Energy Demand Forecasting project.
"""

# Data Configuration
DATA_CONFIG = {
    "raw_data_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
    "raw_data_file": "data/raw/household_power_consumption.csv",
    "processed_data_file": "data/processed/household_power_consumption_processed.csv",
    "target_column": "Global_active_power",
    "resample_frequency": "h",  # hourly
}

# Model Configuration
MODEL_CONFIG = {
    "sequence_length": 24,  # hours of data to predict next hour
    "train_split": 0.8,  # 80% for training
    "test_split": 0.2,  # 20% for testing
    "epochs": 50,
    "batch_size": 64,
    "patience": 10,  # early stopping patience
    "model_file": "models/lstm_model.h5",
    "scaler_file": "models/scaler.pkl",
}

# LSTM Architecture Configuration
LSTM_CONFIG = {
    "lstm_units_1": 50,
    "lstm_units_2": 50,
    "dropout_rate": 0.2,
    "dense_units": 25,
    "optimizer": "adam",
    "loss": "mean_squared_error",
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "Energy Demand Forecasting API",
    "description": "API for energy demand forecasting using LSTM",
    "version": "1.0.0",
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "update_interval": 5000,  # milliseconds
    "max_data_points": 50,  # maximum points to show on chart
    "chart_height": 400,  # pixels
}

# File Paths
PATHS = {
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "models_dir": "models",
    "dashboard_dir": "src/dashboard",
    "templates_dir": "src/dashboard/templates",
    "static_dir": "src/dashboard/static",
}