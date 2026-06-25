
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_sequences(data, seq_length):
    """Creates sequences from the time series data."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def verify_model_performance():
    """
    Loads the trained model, makes predictions on the test set, 
    and evaluates the model's performance.
    """
    # --- 1. Load Model and Scaler ---
    print("Loading model and scaler...")
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")

    # --- 2. Load Data ---
    print("Loading data...")
    processed_data_path = "data/processed/household_power_consumption_processed.csv"
    df = pd.read_csv(processed_data_path, index_col='datetime', parse_dates=True)
    data = df['Global_active_power'].values.reshape(-1, 1)

    # --- 3. Scale and Create Test Set ---
    scaled_data = scaler.transform(data)
    SEQ_LENGTH = 24
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    train_size = int(len(X) * 0.8)
    X_test, y_test = X[train_size:], y[train_size:]

    # --- 4. Make Predictions ---
    print("Making predictions on the test set...")
    y_pred_scaled = model.predict(X_test)

    # --- 5. Inverse Transform ---
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    # --- 6. Calculate Error Metrics ---
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    print(f"\n--- Model Performance on Test Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # --- 7. Generate and Save Plot ---
    print("\nGenerating and saving prediction plot...")
    plt.figure(figsize=(15, 7))
    plot_range = slice(train_size + SEQ_LENGTH, train_size + SEQ_LENGTH + 168)
    plt.plot(df.index[plot_range], y_pred[:168], label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.plot(df.index[plot_range], y_test_actual[:168], label='Actual', color='blue', alpha=0.7, linewidth=1)
    plt.title('Energy Demand: Actual vs. Predicted (1 Week)')
    plt.xlabel('Time')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_vs_actual.png')
    print("Plot saved as prediction_vs_actual.png")

if __name__ == "__main__":
    verify_model_performance()
