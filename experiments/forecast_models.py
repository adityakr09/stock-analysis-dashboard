"""
forecast_combo.py
Combined forecasting: Prophet + ARIMA ensemble

Usage (examples):

# using a cleaned file path directly (one of your uploaded files)
python forecast_combo.py --path "/mnt/data/MM_NS_cleaned.csv" --horizon 30

# using cleaned folder + symbol
python forecast_combo.py --symbol MM_NS --horizon 30

Outputs saved to `outputs/`:
- <symbol>_forecast_prophet.csv
- <symbol>_forecast_arima.csv
- <symbol>_forecast_ensemble.csv
- <symbol>_forecast_plot.png
- <symbol>_model_diagnostics.png (optional)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
CLEANED_DIR = "cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cleaned_file(path=None, symbol=None):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
    else:
        if not symbol:
            raise ValueError("Provide either --path or --symbol")
        candidate = os.path.join(CLEANED_DIR, f"{symbol}_cleaned.csv")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"File not found: {candidate}")
        df = pd.read_csv(candidate)
    # Normalize columns and ensure Date & Close
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if "Close" not in df.columns:
        # try Price or Adj Close
        for alt in ["Price", "Adj Close", "AdjClose"]:
            if alt in df.columns:
                df["Close"] = pd.to_numeric(df[alt], errors="coerce")
                break
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

def prepare_prophet_df(df):
    prophet_df = pd.DataFrame({"ds": df["Date"], "y": df["Close"]})
    return prophet_df

def fit_prophet(prophet_df, yearly_seasonality=False):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=yearly_seasonality)
    m.fit(prophet_df)
    return m

def forecast_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    # keep only ds and yhat, yhat_lower, yhat_upper
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds":"Date"})

def fit_arima(series, max_p=5, max_q=5, max_P=2, max_Q=2, seasonal=False):
    # Use pmdarima.auto_arima to find a good model quickly
    model = pm.auto_arima(series, start_p=1, start_q=1, max_p=max_p, max_q=max_q,
                          seasonal=seasonal, m=1, trace=False, error_action='ignore',
                          suppress_warnings=True, stepwise=True)
    return model

def forecast_arima(model, periods):
    fc, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    return fc, conf_int

def ensemble_forecasts(prophet_df, prophet_fc, arima_fc, periods):
    # prophet_fc: DataFrame with Date and yhat; arima_fc: numpy array aligned with future dates
    # Create index for next periods
    last_date = prophet_df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    prophet_future = prophet_fc.tail(periods).reset_index(drop=True)
    arima_series = pd.Series(arima_fc)
    ensemble = pd.DataFrame({
        "Date": future_dates,
        "Prophet": prophet_future["yhat"].values,
        "ARIMA": arima_series.values
    })
    ensemble["Ensemble"] = ensemble[["Prophet","ARIMA"]].mean(axis=1)
    return ensemble

def evaluate_holdout(df, prophet_model=None, arima_model=None, holdout_days=30):
    if holdout_days >= len(df):
        print("Holdout too large, skipping evaluation")
        return {}
    train = df.iloc[:-holdout_days]
    test = df.iloc[-holdout_days:]
    # Prophet
    prophet_train = prepare_prophet_df(train)
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(prophet_train)
    future = m.make_future_dataframe(periods=holdout_days)
    fc = m.predict(future)
    prophet_pred = fc.tail(holdout_days)["yhat"].values
    # ARIMA
    arima_model_local = pm.auto_arima(train["Close"], seasonal=False, stepwise=True, suppress_warnings=True)
    arima_fc, _ = arima_model_local.predict(n_periods=holdout_days, return_conf_int=True)
    # metrics
    y_true = test["Close"].values
    metrics = {
        "prophet_mae": mean_absolute_error(y_true, prophet_pred),
        "prophet_rmse": mean_squared_error(y_true, prophet_pred, squared=False),
        "arima_mae": mean_absolute_error(y_true, arima_fc),
        "arima_rmse": mean_squared_error(y_true, arima_fc, squared=False)
    }
    return metrics

def save_csv(df, path):
    df.to_csv(path, index=False)
    print("Saved:", path)

def plot_forecasts(df, prophet_full, ensemble_df, symbol):
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Close"], label="Historical", color="black")
    # Prophet full predictions (yhat) aligned by date
    pf = prophet_full[["Date","yhat"]].dropna()
    plt.plot(pf["Date"], pf["yhat"], label="Prophet (yhat)", color="tab:blue", alpha=0.8)
    # ensemble future
    plt.plot(ensemble_df["Date"], ensemble_df["Ensemble"], label="Ensemble", color="tab:green", linestyle="--")
    plt.fill_between(pf["Date"], prophet_full["yhat_lower"], prophet_full["yhat_upper"], color="tab:blue", alpha=0.1)
    plt.title(f"{symbol} — Historical + Forecast (Ensemble)")
    plt.xlabel("Date") 
    plt.ylabel("Price")
    plt.legend()  
    out_png = os.path.join(OUTPUT_DIR, f"{symbol}_forecast_plot.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("Saved plot:", out_png)
    return out_png

def main(args):
    if args.path:
        path = args.path
        df = load_cleaned_file(path=path)
        symbol = os.path.basename(path).replace("_cleaned.csv","").split(".")[0]
    else:
        symbol = args.symbol
        df = load_cleaned_file(symbol=symbol)

    if df.empty:
        raise SystemExit("No data loaded")

    horizon = int(args.horizon)

    # Optional evaluation on holdout
    metrics = {}
    if args.holdout:
        metrics = evaluate_holdout(df, holdout_days=int(args.holdout))
        print("Holdout metrics:", metrics)

    # Prophet preparation and fit
    prophet_df = prepare_prophet_df(df)
    prophet_model = fit_prophet(prophet_df, yearly_seasonality=False)
    prophet_full = forecast_prophet(prophet_model, periods=horizon)

    # Fit ARIMA on the historical Close series
    arima_model = fit_arima(df["Close"], seasonal=False)
    arima_fc, arima_conf = forecast_arima(arima_model, periods=horizon)

    # Build ensemble predictions (aligned future dates)
    ensemble_df = ensemble_forecasts(prophet_df, prophet_full, arima_fc, horizon)

    # Save individual forecasts
    prophet_out = os.path.join(OUTPUT_DIR, f"{symbol}_forecast_prophet.csv")
    arima_out = os.path.join(OUTPUT_DIR, f"{symbol}_forecast_arima.csv")
    ensemble_out = os.path.join(OUTPUT_DIR, f"{symbol}_forecast_ensemble.csv")

    save_csv(prophet_full.tail(horizon), prophet_out)
    arima_dates = ensemble_df[["Date"]].copy()
    arima_df = pd.concat([arima_dates, pd.DataFrame({"arima": arima_fc, "arima_lower": arima_conf[:,0], "arima_upper": arima_conf[:,1]})], axis=1)
    save_csv(arima_df, arima_out)
    save_csv(ensemble_df, ensemble_out)

    # Plot and save
    plot_forecasts(df, prophet_full, ensemble_df, symbol)

    # Print summary
    print("\nForecast complete for", symbol)
    print("Outputs:", prophet_out, arima_out, ensemble_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to cleaned CSV (you can use /mnt/data/MM_NS_cleaned.csv)")
    parser.add_argument("--symbol", type=str, help="Symbol name (uses cleaned/<symbol>_cleaned.csv)")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--holdout", type=int, default=0, help="If >0, use last N days as holdout to evaluate models")
    args = parser.parse_args()
    main(args)
