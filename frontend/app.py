import os, json, requests, pandas as pd, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Energy Intelligence | Forecasting",page_icon="⚡",layout="wide")
API=os.getenv("ENERGY_API_URL","http://127.0.0.1:8000").rstrip()
st.markdown("""
<style>
.block-container {padding-top:1.5rem; padding-bottom:2rem}
[data-testid="stMetric"] {background:rgba(120,140,160,.10);border:1px solid rgba(120,140,160,.22);padding:14px;border-radius:14px}
.hero {padding:22px 24px;border-radius:18px;background:linear-gradient(120deg,#0f766e22,#2563eb22);border:1px solid rgba(120,140,160,.25);margin-bottom:18px}
.small-muted {color:var(--text-color);opacity:.72;font-size:.9rem}
</style>
""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚡ Energy Intelligence</h1><div class="small-muted">Forecasting • Model monitoring • Consumption analytics</div></div>',unsafe_allow_html=True)

def json_bytes(obj):
    return json.dumps(obj, indent=2).encode('utf-8')

def api_get(path,timeout=15):
    r=requests.get(f"{API}{path}",timeout=timeout); r.raise_for_status(); return r.json()
def api_post(path,payload=None,timeout=180):
    r=requests.post(f"{API}{path}",json=payload,timeout=timeout); r.raise_for_status(); return r.json()

with st.sidebar:
    st.title("Control Center")
    st.caption(f"Backend: {API}")
    horizon=st.select_slider("Forecast horizon",options=[6,12,24,48,72,120,168],value=24,format_func=lambda x:f"{x} hours")
    tariff=st.number_input("Energy tariff (₹ / kWh)",min_value=0.0,value=8.5,step=0.5)
    carbon=st.number_input("Emission factor (kg CO₂ / kWh)",min_value=0.0,value=0.82,step=0.05,format="%.2f")
    st.divider()
    if st.button("Train / refresh models",use_container_width=True):
        try:
            with st.spinner("Training and comparing four models..."):
                result=api_post("/train")
            st.session_state["metrics"]=result["metrics"]
            st.success("Training complete.")
        except Exception as e: st.error(f"Training failed: {e}")
    if st.button("Generate forecast",type="primary",use_container_width=True):
        try:
            with st.spinner("Generating forecast..."):
                result=api_post("/forecast",{"horizon":int(horizon)})
            st.session_state["forecast"]=pd.DataFrame(result["forecast"])
            st.success("Forecast ready.")
        except Exception as e: st.error(f"Forecast failed: {e}")
    if st.button("Refresh data summary",use_container_width=True):
        try: st.session_state["summary"]=api_get("/data/summary")
        except Exception as e: st.error(f"API error: {e}")

try:
    health=api_get("/health")
    status="Online" if health.get("status")=="ok" else "Unavailable"
except Exception:
    status="Offline"

h1,h2,h3=st.columns([1,1,2])
h1.metric("API status",status)
h2.metric("Model status","Trained" if status=="Online" and health.get("model_trained") else "Not trained")
h3.markdown("Use the sidebar to train models or generate a forecast. Synthetic data is used until you upload your own hourly CSV.")

tabs=st.tabs(["📊 Overview","🔮 Forecast lab","🧠 Model performance","📁 Data explorer","ℹ️ Methodology"])

with tabs[0]:
    st.subheader("Operational overview")
    try:
        summary=st.session_state.get("summary") or api_get("/data/summary")
        st.session_state["summary"]=summary
        a,b,c,d=st.columns(4)
        a.metric("Hourly records",f"{summary['rows']:,}")
        b.metric("Mean consumption",f"{summary['mean_kwh']:.2f} kWh")
        c.metric("Peak observed",f"{summary['max_kwh']:.2f} kWh")
        d.metric("Observed range",f"{summary['min_kwh']:.1f}–{summary['max_kwh']:.1f}")
        st.caption(f"Data range: {summary['start']} to {summary['end']}")
    except Exception as e:
        st.warning(f"Could not load dataset summary: {e}")
    img=Path(__file__).resolve().parents[1]/"images"/"energy_history.png"
    if img.exists(): st.image(str(img),caption="Historical hourly consumption")
    st.markdown("**Use cases:** demand planning, peak-load awareness, energy cost estimation and facility analytics. This is a decision-support demo, not an automated control system.")

with tabs[1]:
    st.subheader("Forecast lab")
    if "forecast" in st.session_state:
        fc=st.session_state["forecast"].copy()
        fc["timestamp"]=pd.to_datetime(fc["timestamp"])
        total=float(fc["predicted_kwh"].sum())
        peak=fc.loc[fc["predicted_kwh"].idxmax()]
        low=float(fc["lower_estimate_kwh"].sum()) if "lower_estimate_kwh" in fc else total
        high=float(fc["upper_estimate_kwh"].sum()) if "upper_estimate_kwh" in fc else total
        a,b,c,d=st.columns(4)
        a.metric("Forecast energy",f"{total:,.1f} kWh")
        b.metric("Estimated cost",f"₹{total*tariff:,.0f}")
        c.metric("Estimated emissions",f"{total*carbon:,.1f} kg CO₂")
        d.metric("Peak forecast",f"{peak['predicted_kwh']:.1f} kWh",delta=str(peak["timestamp"]))
        chart=fc.set_index("timestamp")[["predicted_kwh"]]
        st.line_chart(chart)
        if {"lower_estimate_kwh","upper_estimate_kwh"}.issubset(fc.columns):
            st.caption("The lower/upper lines in the table are approximate residual bands from holdout errors, not calibrated confidence intervals.")
        st.dataframe(fc,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download forecast CSV",fc.to_csv(index=False).encode(),file_name="energy_forecast.csv",mime="text/csv")
    else:
        st.info("Generate a forecast using the sidebar to populate this page.")
    st.markdown("Cost and emissions are rough estimates calculated from your editable assumptions, not audited bills or emissions reports.")

with tabs[2]:
    st.subheader("Model performance and comparison")
    try:
        metrics=st.session_state.get("metrics") or api_get("/metrics")
        st.session_state["metrics"]=metrics
        a,b,c=st.columns(3)
        a.metric("Selected model",metrics.get("selected_model","—"))
        b.metric("MAE",f"{metrics.get('mae_kwh',float('nan')):.3f} kWh")
        c.metric("RMSE",f"{metrics.get('rmse_kwh',float('nan')):.3f} kWh")
        st.metric("R² score",f"{metrics.get('r2',float('nan')):.3f}")
        comparison=metrics.get("model_comparison",{})
        if comparison:
            table=pd.DataFrame([{"Model":name,**vals} for name,vals in comparison.items()]).sort_values("rmse_kwh")
            st.dataframe(table,use_container_width=True,hide_index=True)
            st.bar_chart(table.set_index("Model")[["mae_kwh","rmse_kwh"]])
        st.caption("Lower MAE/RMSE is better; R² can be negative and is not an accuracy percentage.")
        img=Path(__file__).resolve().parents[1]/"images"/"feature_importance.png"
        if img.exists(): st.image(str(img),caption="Feature importance for tree-based selected model, when available")
        st.download_button("⬇ Download metrics JSON",json_bytes(metrics),file_name="metrics.json",mime="application/json")
    except Exception:
        st.info("Train the models first to display comparison metrics.")

with tabs[3]:
    st.subheader("Data explorer")
    st.markdown("Upload a CSV with `timestamp` and `energy_kwh` columns. Use hourly observations and at least 240 valid rows. Uploading replaces the local demo dataset.")
    uploaded=st.file_uploader("Choose energy consumption CSV",type=["csv"])
    if uploaded is not None:
        try:
            preview=pd.read_csv(uploaded)
            st.write("Preview"); st.dataframe(preview.head(20),use_container_width=True)
            if st.button("Upload dataset to backend"):
                rows=preview[["timestamp","energy_kwh"]].dropna().to_dict(orient="records")
                result=api_post("/data/upload",{"rows":rows},timeout=60)
                st.success(f"Saved {result['rows_saved']} rows. Now train/refresh models.")
        except Exception as e: st.error(f"Could not read/upload CSV: {e}")
    try:
        csv_path=Path(__file__).resolve().parents[1]/"data"/"energy_consumption.csv"
        if csv_path.exists():
            df=pd.read_csv(csv_path)
            st.caption(f"Local CSV preview • {len(df):,} rows")
            st.dataframe(df.head(30),use_container_width=True,hide_index=True)
            st.download_button("⬇ Download current dataset",csv_path.read_bytes(),file_name="energy_consumption.csv",mime="text/csv")
    except Exception as e: st.caption(f"Dataset preview unavailable: {e}")

with tabs[4]:
    st.subheader("How the system works")
    st.markdown("""
    1. **Ingest:** read a CSV with timestamps and energy in kWh.
    2. **Clean:** parse timestamps, remove invalid/negative readings, regularize hourly intervals and interpolate only short gaps.
    3. **Engineer features:** cyclic time features, lags (1 hour to 1 week), rolling means, rolling variability and rolling peak.
    4. **Compare:** Random Forest, Extra Trees, HistGradientBoosting and Ridge regression on a chronological holdout.
    5. **Select:** choose the model with the lowest holdout RMSE, save metrics and diagnostic plots.
    6. **Forecast:** recursively estimate the next 1–168 hourly values.
    """)
    st.warning("Prototype limitations: synthetic data is not representative of a real site; recursive errors accumulate; no authentication or production security is configured; cost and CO₂ figures depend entirely on the assumptions entered.")
