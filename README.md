# AI-Powered Energy Consumption Forecasting System

An end-to-end energy analytics prototype with time-series feature engineering, comparison of four ML models, recursive hourly forecasting, a FastAPI backend, and a multi-tab Streamlit dashboard.

> **Transparency:** If no input CSV is present, synthetic data is generated. Synthetic results are for software demonstration only and do not prove real-world energy forecasting accuracy.

## Features
- CSV ingestion, timestamp validation, duplicate removal and hourly resampling
- Cyclical calendar features, 1/2/3-hour, 24/48-hour and 168-hour lags
- Rolling means, rolling standard deviation and rolling maximum
- Compare Random Forest, Extra Trees, HistGradientBoosting and Ridge Regression
- Chronological holdout evaluation with MAE, RMSE and R²
- Automatic selection by holdout RMSE
- Recursive forecast from 1 to 168 hours, approximate residual band
- Feature importance chart for supported tree models
- FastAPI endpoints and interactive Swagger documentation
- Streamlit dashboard with Overview, Forecast Lab, Model Performance, Data Explorer and Methodology tabs
- Editable tariff and emissions assumptions; cost and CO₂ estimates
- CSV upload, data preview, CSV export and metrics download

## Architecture
```text
CSV / synthetic dataset
  -> validation + hourly resampling
  -> cyclical calendar + lag + rolling features
  -> chronological train/test split
  -> compare 4 regressors -> select lowest holdout RMSE
  -> metrics, comparison table and diagnostic plots
  -> recursive forecast + approximate residual band
  -> FastAPI -> Streamlit dashboard
```

## Structure
```text
AI-Energy-Forecasting/
├── data/                 # Input CSV; generated demo data
├── notebooks/            # Optional exploration notebooks
├── src/pipeline.py       # Data, features, model comparison and forecasting
├── backend/main.py       # REST API
├── frontend/app.py       # Advanced multi-tab dashboard
├── models/               # Saved selected model
├── outputs/              # Metrics, comparison, predictions, forecast
├── images/               # Generated charts and screenshots
├── docs/                 # Simulation and GitHub plans
├── main.py
├── requirements.txt
└── README.md
```

## Dataset
Place `data/energy_consumption.csv` with at least 240 hourly rows:
```csv
timestamp,energy_kwh
2025-01-01 00:00:00,72.5
2025-01-01 01:00:00,69.8
```
These are illustrative values, not measured readings. Required columns: `timestamp`, `energy_kwh`. The dashboard Data Explorer can upload a CSV to the local API.

## Install

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run CLI
```bash
python main.py
python main.py --horizon 72
python main.py --train-only
```
Outputs include `outputs/metrics.json`, `outputs/model_comparison.csv`, `outputs/predictions.csv`, `outputs/forecast.csv`, and charts under `images/`.

## Run backend
Terminal 1:
```bash
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000/docs.

Endpoints:
- `GET /health`
- `GET /data/summary`
- `POST /train`
- `GET /metrics`
- `POST /forecast` with `{"horizon":24}`
- `POST /data/upload` with `{"rows":[{"timestamp":"2025-01-01 00:00:00","energy_kwh":72.5}, ...]}`

## Run frontend
Terminal 2, same virtual environment:
```bash
streamlit run frontend/app.py
```
Open the URL Streamlit prints, usually http://localhost:8501. Keep the API terminal running.

## Metrics and interpretation
- **MAE:** average absolute error in kWh.
- **RMSE:** root mean squared error in kWh; penalizes large errors.
- **R²:** relative fit statistic, not an accuracy percentage; can be negative.
- **Approximate residual band:** based on a holdout residual quantile. It is a diagnostic display, not a calibrated confidence interval.

The model with the lowest chronological holdout RMSE is selected from four candidates. For robust assessment, add rolling-origin backtesting and compare seasonal-naive baselines. Avoid reporting metrics from synthetic data as real-world performance.

## Industry relevance and limitations
The prototype demonstrates concepts useful in demand planning, peak-load awareness, facility scheduling and energy analytics. It does not control equipment or connect to live meters. It has no authentication or production deployment hardening. Validate with real site data before operational use.

## Suggested screenshots
`images/dashboard.png`, `images/model_comparison.png`, `images/actual_vs_predicted.png`, `images/forecast_next_hours.png`, `images/feature_importance.png`, `images/api_docs.png`, `images/github_repo.png`.

## GitHub
Repository suggestion: `AI-Energy-Forecasting`
Description: `Hourly energy forecasting with model comparison, FastAPI and Streamlit analytics dashboard.`
Topics: `machine-learning`, `time-series`, `energy-forecasting`, `python`, `scikit-learn`, `fastapi`, `streamlit`.
