from pathlib import Path
import sys, json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import load_data, train_and_evaluate, forecast_next_hours, MODEL, DATA

app=FastAPI(title="AI Energy Forecasting API",version="2.0.0",
 description="Energy analytics portfolio prototype with model comparison and forecasting.")

class ForecastRequest(BaseModel):
    horizon:int=Field(default=24,ge=1,le=168)

class DatasetRequest(BaseModel):
    rows:list[dict]

@app.get("/")
def root(): return {"service":"AI Energy Forecasting API","version":"2.0.0","docs":"/docs","health":"/health"}
@app.get("/health")
def health(): return {"status":"ok","model_trained":MODEL.exists()}
@app.post("/train")
def train():
    try: return {"message":"Training completed","metrics":train_and_evaluate(load_data())}
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.get("/metrics")
def metrics():
    p=ROOT/"outputs"/"metrics.json"
    if not p.exists(): raise HTTPException(status_code=404,detail="Call POST /train first.")
    return json.loads(p.read_text(encoding="utf-8"))
@app.get("/data/summary")
def data_summary():
    try:
        df=load_data()
        return {"rows":len(df),"start":str(df.timestamp.min()),"end":str(df.timestamp.max()),
                "mean_kwh":float(df.energy_kwh.mean()),"max_kwh":float(df.energy_kwh.max()),
                "min_kwh":float(df.energy_kwh.min()),"missing_values":int(df.isna().sum().sum())}
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.post("/data/upload")
def upload_dataset(req:DatasetRequest):
    try:
        df=pd.DataFrame(req.rows)
        if not {"timestamp","energy_kwh"}.issubset(df.columns):
            raise ValueError("Uploaded rows need timestamp and energy_kwh fields.")
        df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
        df["energy_kwh"]=pd.to_numeric(df["energy_kwh"],errors="coerce")
        df=df.dropna(subset=["timestamp","energy_kwh"])
        if len(df)<240: raise ValueError("Upload at least 240 hourly rows.")
        df.to_csv(DATA,index=False)
        if MODEL.exists(): MODEL.unlink()
        return {"message":"Dataset saved. Train the model to refresh predictions.","rows_saved":len(df)}
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.post("/forecast")
def forecast(req:ForecastRequest):
    try:
        df=load_data()
        if not MODEL.exists(): train_and_evaluate(df)
        fc=forecast_next_hours(df,req.horizon)
        return {"horizon":req.horizon,"unit":"kWh per hour","forecast":fc.to_dict(orient="records")}
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
