"""Enhanced energy forecasting pipeline: model comparison, diagnostics and operational analytics."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "energy_consumption.csv"
MODEL = ROOT / "models" / "energy_model.joblib"
OUT = ROOT / "outputs"
IMG = ROOT / "images"
FEATURES = [
    "hour","day_of_week","day_of_year","month","week_of_year","is_weekend",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "lag_1","lag_2","lag_3","lag_24","lag_48","lag_168",
    "rolling_mean_6","rolling_mean_24","rolling_mean_168",
    "rolling_std_24","rolling_max_24"
]

def ensure_dirs():
    for p in [DATA.parent, MODEL.parent, OUT, IMG]:
        p.mkdir(parents=True, exist_ok=True)

def generate_demo_dataset(days=240, seed=42):
    """Synthetic data with daily/weekly/seasonal patterns; not real meter readings."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days*24, freq="h")
    hour, dow, doy = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.dayofyear.to_numpy()
    weekend = (dow >= 5).astype(int)
    daily = 14*np.sin(2*np.pi*(hour-7)/24)+18
    work = np.where((hour>=8)&(hour<=18)&(dow<5), 24, 5)
    seasonal = 9*np.cos(2*np.pi*doy/365.25)
    trend = np.linspace(0,5,len(ts))
    noise = rng.normal(0,3.2,len(ts))
    energy = np.maximum(8,75+daily+work+seasonal+trend-weekend*9+noise)
    return pd.DataFrame({"timestamp":ts,"energy_kwh":np.round(energy,3)})

def load_data():
    ensure_dirs()
    if not DATA.exists():
        generate_demo_dataset().to_csv(DATA,index=False)
        print(f"[data] Created synthetic demo dataset: {DATA}")
    df = pd.read_csv(DATA)
    if not {"timestamp","energy_kwh"}.issubset(df.columns):
        raise ValueError("CSV must contain columns timestamp,energy_kwh.")
    df["timestamp"] = pd.to_datetime(df["timestamp"],errors="coerce")
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"],errors="coerce")
    df = df.dropna(subset=["timestamp","energy_kwh"])
    df = df[df.energy_kwh>=0].drop_duplicates("timestamp",keep="last").sort_values("timestamp")
    if len(df)<240:
        raise ValueError("At least 240 hourly rows (10 days) are required.")
    df = df.set_index("timestamp").resample("h").mean()
    df["energy_kwh"] = df["energy_kwh"].interpolate(limit=3,limit_direction="both")
    return df.dropna().reset_index()

def add_features(df):
    d=df.sort_values("timestamp").copy()
    t=d["timestamp"]
    d["hour"]=t.dt.hour
    d["day_of_week"]=t.dt.dayofweek
    d["day_of_year"]=t.dt.dayofyear
    d["month"]=t.dt.month
    d["week_of_year"]=t.dt.isocalendar().week.astype(int)
    d["is_weekend"]=(t.dt.dayofweek>=5).astype(int)
    d["hour_sin"]=np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"]=np.cos(2*np.pi*d["hour"]/24)
    d["dow_sin"]=np.sin(2*np.pi*d["day_of_week"]/7)
    d["dow_cos"]=np.cos(2*np.pi*d["day_of_week"]/7)
    d["month_sin"]=np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]=np.cos(2*np.pi*d["month"]/12)
    for lag in [1,2,3,24,48,168]:
        d[f"lag_{lag}"]=d["energy_kwh"].shift(lag)
    shifted=d["energy_kwh"].shift(1)
    d["rolling_mean_6"]=shifted.rolling(6).mean()
    d["rolling_mean_24"]=shifted.rolling(24).mean()
    d["rolling_mean_168"]=shifted.rolling(168).mean()
    d["rolling_std_24"]=shifted.rolling(24).std()
    d["rolling_max_24"]=shifted.rolling(24).max()
    return d.dropna(subset=FEATURES+["energy_kwh"]).reset_index(drop=True)

def _score(y, pred):
    return {
        "mae_kwh":float(mean_absolute_error(y,pred)),
        "rmse_kwh":float(np.sqrt(mean_squared_error(y,pred))),
        "r2":float(r2_score(y,pred))
    }

def train_and_evaluate(df):
    d=add_features(df)
    split=int(len(d)*0.8)
    train,test=d.iloc[:split],d.iloc[split:]
    Xtr,ytr=train[FEATURES],train["energy_kwh"]
    Xte,yte=test[FEATURES],test["energy_kwh"]
    candidates={
        "Random Forest":RandomForestRegressor(n_estimators=250,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Extra Trees":ExtraTreesRegressor(n_estimators=250,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Hist Gradient Boosting":HistGradientBoostingRegressor(max_iter=180,learning_rate=0.08,max_leaf_nodes=20,l2_regularization=1.0,random_state=42),
        "Ridge Regression":Ridge(alpha=2.0)
    }
    comparison={}
    fitted={}
    for name,model in candidates.items():
        model.fit(Xtr,ytr)
        p=model.predict(Xte)
        comparison[name]=_score(yte,p)
        fitted[name]=model
    # Select lowest holdout RMSE. Keep all comparison results transparent.
    winner=min(comparison,key=lambda name:comparison[name]["rmse_kwh"])
    model=fitted[winner]
    pred=model.predict(Xte)
    # Empirical absolute residual quantile used as a simple diagnostic band, not a calibrated guarantee.
    residual=np.abs(yte.to_numpy()-pred)
    band=float(np.quantile(residual,0.90))
    metrics={
        "selected_model":winner,
        "model_comparison":comparison,
        "train_rows":int(len(train)),
        "test_rows":int(len(test)),
        **_score(yte,pred),
        "approx_90pct_abs_residual_kwh":band,
        "test_start":str(test.timestamp.min()),
        "test_end":str(test.timestamp.max()),
        "features":FEATURES,
        "note":"Synthetic-data results are demonstration-only. Residual bands are approximate diagnostics, not guaranteed prediction intervals."
    }
    joblib.dump({"model":model,"features":FEATURES,"selected_model":winner,"residual_band":band},MODEL)
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    pd.DataFrame({"timestamp":test.timestamp,"actual_kwh":yte.to_numpy(),"predicted_kwh":pred,
                  "error_kwh":yte.to_numpy()-pred}).to_csv(OUT/"predictions.csv",index=False)
    pd.DataFrame([{"model":k,**v} for k,v in comparison.items()]).to_csv(OUT/"model_comparison.csv",index=False)
    _save_plots(df,test,pred,model,band)
    print("\n=== Model comparison (chronological holdout) ===")
    for name,values in comparison.items():
        print(f"{name}: MAE={values['mae_kwh']:.3f} | RMSE={values['rmse_kwh']:.3f} | R2={values['r2']:.3f}")
    print(f"Selected model by holdout RMSE: {winner}")
    return metrics

def _save_plots(df,test,pred,model,band):
    IMG.mkdir(exist_ok=True)
    plt.figure(figsize=(12,4)); plt.plot(df.timestamp,df.energy_kwh,lw=.8)
    plt.title("Hourly Energy Consumption"); plt.xlabel("Timestamp"); plt.ylabel("Energy (kWh)")
    plt.tight_layout(); plt.savefig(IMG/"energy_history.png",dpi=150); plt.close()
    plt.figure(figsize=(12,4)); plt.plot(test.timestamp,test.energy_kwh,label="Actual",lw=1)
    plt.plot(test.timestamp,pred,label="Predicted",lw=1)
    plt.title("Actual vs Predicted — Holdout"); plt.xlabel("Timestamp"); plt.ylabel("kWh")
    plt.legend(); plt.tight_layout(); plt.savefig(IMG/"actual_vs_predicted.png",dpi=150); plt.close()
    plt.figure(figsize=(8,4)); plt.hist(test.energy_kwh.to_numpy()-pred,bins=30)
    plt.title("Prediction Error Distribution"); plt.xlabel("Actual − predicted (kWh)")
    plt.tight_layout(); plt.savefig(IMG/"error_distribution.png",dpi=150); plt.close()
    if hasattr(model,"feature_importances_"):
        imp=pd.Series(model.feature_importances_,index=FEATURES).sort_values().tail(15)
        plt.figure(figsize=(9,6)); imp.plot(kind="barh"); plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance"); plt.tight_layout(); plt.savefig(IMG/"feature_importance.png",dpi=150); plt.close()

def forecast_next_hours(df,horizon=24):
    if not 1<=horizon<=168: raise ValueError("horizon must be between 1 and 168.")
    if not MODEL.exists(): raise FileNotFoundError("Train first.")
    bundle=joblib.load(MODEL); model=bundle["model"]; band=float(bundle.get("residual_band",0))
    hist=df.sort_values("timestamp").reset_index(drop=True)
    vals=hist.energy_kwh.astype(float).tolist(); last=hist.timestamp.max(); rows=[]
    for step in range(1,horizon+1):
        ts=last+pd.Timedelta(hours=step)
        if len(vals)<168: raise ValueError("Need at least 168 hourly observations.")
        row={
            "hour":ts.hour,"day_of_week":ts.dayofweek,"day_of_year":ts.dayofyear,"month":ts.month,
            "week_of_year":int(ts.isocalendar().week),"is_weekend":int(ts.dayofweek>=5),
            "hour_sin":np.sin(2*np.pi*ts.hour/24),"hour_cos":np.cos(2*np.pi*ts.hour/24),
            "dow_sin":np.sin(2*np.pi*ts.dayofweek/7),"dow_cos":np.cos(2*np.pi*ts.dayofweek/7),
            "month_sin":np.sin(2*np.pi*ts.month/12),"month_cos":np.cos(2*np.pi*ts.month/12),
            "lag_1":vals[-1],"lag_2":vals[-2],"lag_3":vals[-3],"lag_24":vals[-24],
            "lag_48":vals[-48],"lag_168":vals[-168],
            "rolling_mean_6":float(np.mean(vals[-6:])),
            "rolling_mean_24":float(np.mean(vals[-24:])),
            "rolling_mean_168":float(np.mean(vals[-168:])),
            "rolling_std_24":float(np.std(vals[-24:],ddof=1)),
            "rolling_max_24":float(np.max(vals[-24:]))
        }
        y=max(0.,float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        vals.append(y)
        rows.append({"timestamp":ts,"predicted_kwh":round(y,3),
                     "lower_estimate_kwh":round(max(0,y-band),3),
                     "upper_estimate_kwh":round(y+band,3)})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"forecast.csv",index=False)
    recent=hist.tail(168)
    plt.figure(figsize=(11,4)); plt.plot(recent.timestamp,recent.energy_kwh,label="Recent actual")
    plt.plot(result.timestamp,result.predicted_kwh,label="Forecast",lw=2)
    plt.fill_between(result.timestamp,result.lower_estimate_kwh,result.upper_estimate_kwh,alpha=.2,label="Approx. residual band")
    plt.title(f"Next {horizon} Hours Forecast"); plt.xlabel("Timestamp"); plt.ylabel("kWh")
    plt.legend(); plt.tight_layout(); plt.savefig(IMG/"forecast_next_hours.png",dpi=150); plt.close()
    return result

def run_pipeline(horizon=24):
    df=load_data()
    metrics=train_and_evaluate(df)
    return {"metrics":metrics,"forecast":forecast_next_hours(df,horizon)}
