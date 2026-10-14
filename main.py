import argparse
from src.pipeline import load_data, train_and_evaluate, forecast_next_hours

def main():
    p=argparse.ArgumentParser(description="AI energy consumption forecasting")
    p.add_argument("--horizon",type=int,default=24,help="Forecast horizon: 1-168 hours")
    p.add_argument("--train-only",action="store_true")
    args=p.parse_args()
    df=load_data()
    train_and_evaluate(df)
    if not args.train_only:
        fc=forecast_next_hours(df,args.horizon)
        print("\nForecast preview:")
        print(fc.head(10).to_string(index=False))
if __name__=="__main__": main()
