import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ---------- PAGE SETTINGS ----------
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📊 Stock Analysis Dashboard")
st.caption("End-to-end stock data analysis using Python, Pandas & Basic Trend Forecasting")

CLEANED_DIR = "cleaned"  # folder where *_cleaned.csv files live

# ---------- HELPER FUNCTIONS ----------
@st.cache_data
def list_symbols():
   
    paths = glob.glob(os.path.join(CLEANED_DIR, "*_cleaned.csv"))
    symbols = {}
    for p in paths:
        name = os.path.basename(p).replace("_cleaned.csv", "")
        symbols[name] = p
    return symbols

@st.cache_data
def load_data(path):
    """Load a cleaned CSV and prepare columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Technical indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility20"] = df["Daily_Return"].rolling(20).std()

    return df

def simple_linear_forecast(df, days=30):
    """Straight-line trend forecast (easy in interview)."""
    hist = df.tail(60) if len(df) > 60 else df.copy()
    x = hist["Date"].map(datetime.toordinal).values
    y = hist["Close"].values
    m, c = np.polyfit(x, y, 1)

    last_date = hist["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)
    x_future = future_dates.map(datetime.toordinal).values
    y_future = m * x_future + c

    return pd.DataFrame({"Date": future_dates, "Predicted_Close": y_future})

# ---------- SIDEBAR ----------
symbols = list_symbols()
if not symbols:
    st.error("❌ No cleaned CSV files found in 'cleaned/' folder!")
    st.stop()

with st.sidebar:
    st.header("Settings")
    chosen_stock = st.selectbox("Select Stock Symbol", list(symbols.keys()))
    show_sma20 = st.checkbox("Show SMA 20", value=True)
    show_sma50 = st.checkbox("Show SMA 50", value=True)
    forecast_days = st.slider("Forecast Days", 7, 60, 30, 1)

# ---------- LOAD AND PROCESS ----------
df = load_data(symbols[chosen_stock])

# ---------- TOP KPIs ----------
latest = df.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Latest Close", f"{latest['Close']:.2f}")
col2.metric("20-Day Return", f"{(df['Close'].iloc[-1]/df['Close'].iloc[-20]-1)*100:.2f}%" if len(df)>20 else "N/A")
col3.metric("20-Day Volatility", f"{latest['Volatility20']*100:.2f}%" if not np.isnan(latest['Volatility20']) else "N/A")

# ---------- PRICE CHART ----------
st.subheader("📈 Price Chart with Moving Averages")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(color="#0078D7")))
if show_sma20: fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], mode="lines", name="SMA 20", line=dict(color="#E36209")))
if show_sma50: fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], mode="lines", name="SMA 50", line=dict(color="#4CAF50")))
fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# ---------- FORECAST ----------
st.subheader("🔮 Simple Trend Forecast")
forecast_df = simple_linear_forecast(df, forecast_days)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Historical Price"))
fig_fc.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"], mode="lines",
                            name="Forecast Trend", line=dict(color="red", dash="dash")))
fig_fc.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_fc, use_container_width=True)

st.dataframe(forecast_df, use_container_width=True)


# ---------- ADVANCED ARIMA FORECAST ----------
st.subheader("📉 ARIMA Forecast (Advanced)")

try:
    import pmdarima as pm

    # Fit ARIMA
    with st.spinner("Training ARIMA model..."):
        arima_model = pm.auto_arima(
            df["Close"],
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True
        )

    # Forecast future prices
    arima_forecast = arima_model.predict(n_periods=forecast_days)

    # Create future dates
    last_date = df["Date"].iloc[-1]
    arima_future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    # Build forecast DataFrame
    arima_df = pd.DataFrame({
        "Date": arima_future_dates,
        "Predicted_Close": arima_forecast
    })
   
    # Plot ARIMA Forecast
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Historical Price"))
    fig_arima.add_trace(go.Scatter(x=arima_df["Date"], y=arima_df["Predicted_Close"], mode="lines",
                                   name="ARIMA Forecast", line=dict(color="#FF5733", dash="dot")))
    fig_arima.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_arima, use_container_width=True)

    # Show data table
    st.dataframe(arima_df)

except Exception as e:
    st.error(f"ARIMA model failed: {e}")
    st.info("Try selecting a stock with more data.")
