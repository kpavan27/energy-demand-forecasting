
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def create_sequences(data, seq_length):
    """Creates sequences from the time series data."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model():
    """Trains the LSTM model and saves it."""
    # --- 1. Load Data from SQLite ---
    print("Loading data for training from SQLite...")
    db_path = "data/energy_data.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT datetime, Global_active_power FROM energy_data', conn, index_col='datetime', parse_dates=['datetime'])
    conn.close()
    data = df['Global_active_power'].values.reshape(-1, 1)

    # --- 2. Scale Data ---
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, 'models/scaler.pkl') # Save the scaler

    # --- 3. Create Sequences ---
    SEQ_LENGTH = 24  # Use 24 hours of data to predict the next hour
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # --- 4. Split Data ---
    # Splitting the data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- 5. Build LSTM Model ---
    print("Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # --- 6. Train Model ---
    print("Training LSTM model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=1)

    # --- 7. Evaluate Model ---
    print("Evaluating model...")
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}")

    # --- 8. Save Model ---
    print("Saving model...")
    model.save("models/lstm_model.h5")

    print("Model training complete.")

if __name__ == "__main__":
    train_lstm_model()
