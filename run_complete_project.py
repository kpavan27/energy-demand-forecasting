# Energy Demand Forecasting: Complete Project Script
#
# This script contains all the code to run the energy demand forecasting project.
# It includes data downloading, preprocessing, model training, and a web dashboard.
#
# To run this project, you need to have the following libraries installed:
#
# tensorflow
# pandas
# scikit-learn
# fastapi
# uvicorn
# numpy
# matplotlib
# requests
# python-dotenv
# statsmodels
# Jinja2
#
# You can install them using pip:
# pip install tensorflow pandas scikit-learn fastapi uvicorn numpy matplotlib requests python-dotenv statsmodels Jinja2

import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
import uvicorn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. Data Downloading and Preprocessing ---
def download_and_process_data():
    """
    Downloads and preprocesses the household power consumption dataset.
    """
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

    # Download the zip file
    print("Downloading data...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Extract and read the data
    print("Extracting and reading data...")
    with z.open("household_power_consumption.txt") as f:
        df = pd.read_csv(f, sep=';', low_memory=False)

    # --- Data Preprocessing ---
    print("Preprocessing data...")

    # 1. Combine Date and Time into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('datetime')
    df = df.drop(['Date', 'Time'], axis=1)

    # 2. Handle missing values (represented as '?')
    df = df.replace('?', np.nan)

    # 3. Convert columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    # 4. Resample to hourly data to make the dataset more manageable
    df_resampled = df.resample('h').mean()

    # 5. Handle missing values in resampled data (e.g., using forward fill)
    df_resampled = df_resampled.ffill()

    # --- Save the data ---
    raw_data_path = "data/raw/household_power_consumption.csv"
    processed_data_path = "data/processed/household_power_consumption_processed.csv"

    print(f"Saving raw data to {raw_data_path}...")
    df.to_csv(raw_data_path)

    print(f"Saving processed data to {processed_data_path}...")
    df_resampled.to_csv(processed_data_path)

    print("Data download and processing complete.")

# --- 2. Model Training ---
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
    """
    Trains the LSTM model and saves it.
    """
    os.makedirs("models", exist_ok=True)
    # --- 1. Load Data ---
    print("Loading data for training...")
    processed_data_path = "data/processed/household_power_consumption_processed.csv"
    df = pd.read_csv(processed_data_path, index_col='datetime', parse_dates=True)
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

# --- 3. Model Verification ---
def verify_model_performance():
    """
    Loads the trained model, makes predictions on the test set, 
    and evaluates the model's performance.
    """
    # --- 1. Load Model and Scaler ---
    print("Loading model and scaler for verification...")
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")

    # --- 2. Load Data ---
    print("Loading data for verification...")
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
    plt.plot(df.index[train_size+SEQ_LENGTH:], y_test_actual, label='Actual', color='blue')
    plt.plot(df.index[train_size+SEQ_LENGTH:], y_pred, label='Predicted', color='red', linestyle='--')
    plt.title('Energy Demand: Actual vs. Predicted')
    plt.xlabel('Time')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_vs_actual.png')
    print("Plot saved as prediction_vs_actual.png")

# --- 4. Web Dashboard ---
app = FastAPI()

def setup_dashboard():
    """Sets up the FastAPI application, templates, and static files."""
    os.makedirs("src/dashboard/templates", exist_ok=True)
    os.makedirs("src/dashboard/static", exist_ok=True)

    # HTML file
    index_html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Energy Demand Forecasting</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">
            <h1>Energy Demand Forecasting</h1>
            <div class="chart-container">
                <canvas id="energy-chart"></canvas>
            </div>
            <div class="info-container">
                <h2>Real-time Monitoring</h2>
                <p>Actual vs. Forecasted Energy Demand</p>
                <div id="prediction-output"></div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <script src="/static/script.js"></script>
    </body>
    </html>
    '''
    with open("src/dashboard/templates/index.html", "w") as f:
        f.write(index_html_content)

    # CSS file
    style_css_content = '''
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #f0f2f5;
        color: #333;
        margin: 0;
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }

    .container {
        background-color: #fff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        width: 80%;
        max-width: 1200px;
        text-align: center;
    }

    h1 {
        color: #1c3d5a;
        margin-bottom: 20px;
    }

    .chart-container {
        position: relative;
        height: 400px;
        width: 100%;
        margin-bottom: 30px;
    }

    .info-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    #prediction-output {
        font-size: 1.2em;
        font-weight: bold;
        color: #1c3d5a;
        margin-top: 10px;
    }
    '''
    with open("src/dashboard/static/style.css", "w") as f:
        f.write(style_css_content)

    # JavaScript file
    script_js_content = '''
    document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('energy-chart').getContext('2d');
    let energyChart;
    let chartData = {
        labels: [],
        datasets: [
            {
                label: 'Actual Energy Demand',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                fill: true,
            },
            {
                label: 'Forecasted Energy Demand',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderWidth: 2,
                fill: true,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'hour',
                    tooltipFormat: 'yyyy-MM-dd HH:mm'
                },
                title: {
                    display: true,
                    text: 'Time'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Global Active Power (kilowatts)'
                }
            }
        }
    };

    energyChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: chartOptions,
    });

    async function getInitialData() {
        try {
            const response = await fetch('/get_initial_data');
            const data = await response.json();
            const now = new Date();

            chartData.labels = Array.from({ length: 24 }, (_, i) => {
                const date = new Date(now);
                date.setHours(now.getHours() - (23 - i));
                return date;
            });

            chartData.datasets[0].data = data.initial_data;
            // Initialize forecast data with nulls
            chartData.datasets[1].data = new Array(data.initial_data.length).fill(null);
            energyChart.update();
            // Start the prediction loop after initial data is loaded
            setInterval(makePrediction, 5000); // Predict every 5 seconds
        } catch (error) {
            console.error('Error fetching initial data:', error);
        }
    }

    async function makePrediction() {
        const sequence = chartData.datasets[0].data.slice(-24);
        try {
            const [predictResponse, actualResponse] = await Promise.all([
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ sequence }),
                }),
                fetch('/get_next_actual')
            ]);

            const predictResult = await predictResponse.json();
            const actualResult = await actualResponse.json();

            if (predictResult.prediction && actualResult.next_actual) {
                const prediction = predictResult.prediction;
                const next_actual = actualResult.next_actual;

                document.getElementById('prediction-output').innerText = `Forecasted Demand: ${prediction.toFixed(2)} kW`;

                // Update the chart
                const now = new Date();
                chartData.labels.push(now);
                chartData.datasets[0].data.push(next_actual);
                chartData.datasets[1].data.push(prediction);

                // To keep the chart from getting too crowded, we'll shift the data
                if (chartData.labels.length > 50) {
                    chartData.labels.shift();
                    chartData.datasets[0].data.shift();
                    chartData.datasets[1].data.shift();
                }

                energyChart.update();
            } else {
                console.error('Prediction error or no more actual data:', predictResult.error || actualResult.error);
            }
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    }

    getInitialData();
});
    '''
    with open("src/dashboard/static/script.js", "w") as f:
        f.write(script_js_content)

    app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")
    templates = Jinja2Templates(directory="src/dashboard/templates")

    try:
        model = load_model("models/lstm_model.h5")
        scaler = joblib.load("models/scaler.pkl")
    except Exception as e:
        print(f"!!! ERROR: Failed to load model or scaler: {e}")
        return

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/predict")
    async def predict(data: dict):
        try:
            sequence = data.get("sequence", [])
            if len(sequence) != 24:
                return {"error": "Sequence must have 24 data points."}

            sequence = np.array(sequence).reshape(-1, 1)
            scaled_sequence = scaler.transform(sequence)
            scaled_sequence = scaled_sequence.reshape(1, 24, 1)
            prediction_scaled = model.predict(scaled_sequence)
            prediction = scaler.inverse_transform(prediction_scaled)
            return {"prediction": float(prediction[0][0])}
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @app.get("/get_initial_data")
    async def get_initial_data():
        df = pd.read_csv("data/processed/household_power_consumption_processed.csv", index_col='datetime', parse_dates=True)
        initial_data = df['Global_active_power'].tail(24).tolist()
        return {"initial_data": initial_data}

    current_data_index = 24
    @app.get("/get_next_actual")
    async def get_next_actual():
        nonlocal current_data_index
        try:
            df = pd.read_csv("data/processed/household_power_consumption_processed.csv", index_col='datetime', parse_dates=True)
            if current_data_index < len(df):
                next_value = df['Global_active_power'].iloc[current_data_index]
                current_data_index += 1
                return {"next_actual": next_value}
            else:
                return {"next_actual": None}
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

def start_server():
    """Starts the FastAPI server."""
    setup_dashboard()
    print("Registered routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            print(f"  {route.path}, methods: {route.methods}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    print("Starting Energy Demand Forecasting Project Setup...")

    # Step 1: Download and process data
    if not os.path.exists("data/processed/household_power_consumption_processed.csv"):
        download_and_process_data()
    else:
        print("Data already downloaded and processed.")

    # Step 2: Train the model
    if not os.path.exists("models/lstm_model.h5"):
        train_lstm_model()
    else:
        print("Model already trained.")

    # Step 3: Verify model performance
    if os.path.exists("models/lstm_model.h5"):
        while True:
            verify = input("Do you want to verify the model performance? (y/n): ").lower()
            if verify in ['y', 'n']:
                break
        if verify == 'y':
            verify_model_performance()

    # Step 4: Start the dashboard server
    while True:
        start = input("Do you want to start the dashboard server? (y/n): ").lower()
        if start in ['y', 'n']:
            break
    if start == 'y':
        print("Starting the dashboard server...")
        print("Access the dashboard at: http://localhost:8000")
        start_server()