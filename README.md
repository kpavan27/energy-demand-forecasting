# Energy demand forecasting

An LSTM that forecasts next-hour household electricity demand from the previous 24 hours, with a FastAPI service and a React dashboard that plots predictions against actual demand as the data streams in.

**Data:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (one house in France, minute-level readings, 2006–2010). It's resampled to hourly means, gaps are forward-filled, and the target is `Global_active_power` (kW).

## How it works

```
download_data.py ──► hourly resample + forward fill ──► SQLite / CSV
                                                          │
train_lstm.py ──► MinMax scaling ──► 24-hour windows ──► 2 × LSTM(50) + dropout ──► models/lstm_model.h5
                                                          │
src/api/main.py (FastAPI) ──► /predict, /get_initial_data, /get_next_actual
                                                          │
frontend/ (React + TypeScript + Vite) and src/dashboard/ ──► live actual-vs-predicted chart
```

- **Split:** chronological 80/20, with no shuffling across time.
- **Training:** Adam on MSE, batch size 64, up to 50 epochs with early stopping (patience 10).
- **Evaluation:** `run_complete_project.py` reports MAE and RMSE in kW on the held-out final 20% and saves an actual-vs-predicted plot.

## Run it

```bash
git clone https://github.com/kpavan27/energy-demand-forecasting.git
cd energy-demand-forecasting
pip install -r requirements.txt
python run_complete_project.py   # download → preprocess → train (if needed) → evaluate → start API on :8000
```

To run the React dashboard:

```bash
cd frontend && npm install && npm run dev
```

Or run the pieces one at a time:

```bash
python src/data_preprocessing/download_data.py
python src/model/train_lstm.py
python verify_model.py
uvicorn src.api.main:app --port 8000
```

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/predict` | Body `{"sequence": [24 hourly values in kW]}` → next-hour prediction |
| `GET` | `/get_initial_data` | First 24 hours, used to seed the dashboard |
| `GET` | `/get_next_actual` | Next actual value, for comparison |

## Project layout

```
src/data_preprocessing/   download, resample and store the data
src/model/                LSTM training
src/api/                  FastAPI service
src/dashboard/            lightweight HTML/JS dashboard served by the API
frontend/                 React + TypeScript dashboard
config.py                 window length, split, epochs and other settings
verify_model.py           standalone evaluation
```

## Known limitations and next steps

- Early stopping currently monitors the test split, and the scaler is fit on the full series. Both leak a little information from the test period, so the reported test error is slightly optimistic. The fix is a separate validation split, with the scaler fit on training data only.
- Report the LSTM against a persistence baseline (next hour = this hour) and a seasonal baseline (same hour yesterday). An hourly forecaster only adds value if it beats those.
- Add calendar features (hour, weekday) and test multi-step horizons.
