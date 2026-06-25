# Energy Demand Forecasting

## Project Description

This project implements an LSTM-based energy demand forecasting system using household power consumption data. The system includes:

- **Data Processing**: Downloads and preprocesses household power consumption data from UCI ML Repository
- **LSTM Model**: Deep learning model for time series forecasting
- **Web Dashboard**: Real-time visualization of predictions vs actual values
- **REST API**: FastAPI-based API for model predictions

The model predicts the next hour's energy demand based on the previous 24 hours of data.

## Features

- ✅ Automated data download and preprocessing
- ✅ LSTM neural network for time series forecasting
- ✅ Real-time web dashboard with interactive charts
- ✅ REST API for model predictions
- ✅ Model performance evaluation and visualization
- ✅ Error handling and data validation

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Energy_Demand_Forecasting
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete project**:
   ```bash
   python run_complete_project.py
   ```

## Usage

### Quick Start

Run the complete project setup:
```bash
python run_complete_project.py
```

This will:
1. Download and preprocess the data (if not already done)
2. Train the LSTM model (if not already trained)
3. Optionally verify model performance
4. Start the web dashboard server

### Individual Components

#### Data Preprocessing
```bash
python src/data_preprocessing/download_data.py
```

#### Model Training
```bash
python src/model/train_lstm.py
```

#### Model Verification
```bash
python verify_model.py
```

#### Start API Server
```bash
python src/api/main.py
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Web Dashboard

Once the server is running, access the dashboard at:
- **URL**: http://localhost:8000
- **Features**: Real-time energy demand forecasting with interactive charts

### API Endpoints

- `GET /` - Web dashboard
- `POST /predict` - Get energy demand prediction
- `GET /get_initial_data` - Get initial 24 hours of data
- `GET /get_next_actual` - Get next actual value for comparison

#### Example API Usage

```python
import requests

# Get prediction
response = requests.post('http://localhost:8000/predict', 
                        json={'sequence': [1.2, 1.5, 1.3, ...]})  # 24 values
prediction = response.json()['prediction']
```

## Project Structure

```
Energy_Demand_Forecasting/
├── data/
│   ├── raw/                           # Raw dataset
│   └── processed/                     # Preprocessed data
├── models/
│   ├── lstm_model.h5                 # Trained LSTM model
│   └── scaler.pkl                    # Data scaler
├── src/
│   ├── api/                          # FastAPI application
│   ├── dashboard/                    # Web dashboard files
│   ├── data_preprocessing/           # Data processing scripts
│   └── model/                        # Model training scripts
├── notebooks/                        # Jupyter notebooks
├── run_complete_project.py           # Main execution script
├── verify_model.py                   # Model verification script
└── requirements.txt                  # Python dependencies
```

## Model Performance

The LSTM model is evaluated using:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of average squared differences

## Requirements

- Python 3.8+
- TensorFlow 2.19.0+
- Pandas 2.0.0+
- FastAPI 0.104.0+
- See `requirements.txt` for complete list

## Troubleshooting

### Common Issues

1. **Model files not found**: Run the training script first
2. **Data files not found**: Run the data preprocessing script first
3. **Port already in use**: Change the port in the server configuration

### Error Messages

- `Model or scaler not loaded`: Ensure model files exist in the `models/` directory
- `Sequence must have 24 data points`: Provide exactly 24 values for prediction
- `Processed data file not found`: Run data preprocessing first

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
