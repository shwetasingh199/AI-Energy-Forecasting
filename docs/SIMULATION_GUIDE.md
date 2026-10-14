# Enhanced simulation guide

1. Run `python main.py` to create synthetic hourly data, compare four models, save evaluation metrics and forecast the next 24 hours.
2. Inspect `outputs/model_comparison.csv` and `outputs/metrics.json`.
3. Review charts in `images/`, including feature importance when the selected model supports it.
4. Start API with `uvicorn backend.main:app --reload`; test endpoints at `http://127.0.0.1:8000/docs`.
5. In another terminal run `streamlit run frontend/app.py`.
6. Explore the tabs, adjust tariff/emissions assumptions, generate forecasts, inspect metrics and download CSVs.
7. In Data Explorer, upload a CSV with timestamp and energy_kwh columns (minimum 240 valid rows); then train/refresh models.

The default dataset is synthetic, with daily, weekly and seasonal patterns, trend and noise. It is not real meter data. Approximate residual bands are not calibrated confidence intervals. Real-world evaluation needs genuine data and rolling-origin backtesting.
