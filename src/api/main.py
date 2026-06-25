from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

# Load model and scaler with error handling
try:
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    print("Model and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    print("Please run the training script first to generate the model files.")
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.post("/predict")
async def predict(data: dict):
    """Receives a sequence of 24 data points and returns a prediction."""
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please ensure model files exist."}
    
    sequence = data.get("sequence", [])
    if len(sequence) != 24:
        return {"error": "Sequence must have 24 data points."}

    try:
        # Validate input data
        if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in sequence):
            return {"error": "All sequence values must be valid numbers."}

        # Scale the input sequence
        sequence = np.array(sequence).reshape(-1, 1)
        scaled_sequence = scaler.transform(sequence)

        # Reshape for the model
        scaled_sequence = scaled_sequence.reshape(1, 24, 1)

        # Make prediction
        prediction_scaled = model.predict(scaled_sequence, verbose=0)

        # Inverse scale the prediction
        prediction = scaler.inverse_transform(prediction_scaled)

        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/get_initial_data")
async def get_initial_data():
    """Returns the last 24 hours of data to initialize the chart."""
    try:
        db_path = "data/energy_data.db"
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('SELECT datetime, Global_active_power FROM energy_data ORDER BY datetime DESC LIMIT 24', conn, index_col='datetime', parse_dates=['datetime'])
        conn.close()
        initial_data = df['Global_active_power'].iloc[::-1].tolist() # Reverse to get chronological order
        return {"initial_data": initial_data}
    except Exception as e:
        return {"error": f"Failed to load initial data from DB: {str(e)}"}

# This global variable will keep track of the current position in the dataset
# to simulate streaming new actual values from the database.
# It should be initialized after the initial data is loaded.
_current_data_offset = 0

@app.get("/get_next_actual")
async def get_next_actual():
    """Returns the next actual value from the dataset, simulating a stream."""
    global _current_data_offset
    try:
        db_path = "data/energy_data.db"
        conn = sqlite3.connect(db_path)
        # Get the total number of records to know our bounds
        total_records_df = pd.read_sql('SELECT COUNT(*) FROM energy_data', conn)
        total_records = total_records_df.iloc[0, 0]

        # Fetch the next record based on the offset
        # We need to order by datetime to ensure we get the next chronological record
        query = f'SELECT Global_active_power FROM energy_data ORDER BY datetime ASC LIMIT 1 OFFSET {_current_data_offset}'
        next_value_df = pd.read_sql(query, conn)
        conn.close()

        if not next_value_df.empty:
            next_value = next_value_df['Global_active_power'].iloc[0]
            _current_data_offset += 1
            return {"next_actual": float(next_value)}
        else:
            return {"next_actual": None, "message": "No more data available"}
    except Exception as e:
        return {"error": f"Failed to get next actual value from DB: {str(e)}"}

# Mount the static files directory for the React frontend
# This should be after all your API routes
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")