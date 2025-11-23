import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CLEANED_FOLDER = "cleaned"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameters
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ---------- INDICATOR HELPERS ----------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

# ---------- PROCESS FILES ----------
metrics = []
cleaned_files = glob.glob(os.path.join(CLEANED_FOLDER, "*_cleaned.csv"))

if not cleaned_files:
    print("No cleaned CSVs found in 'cleaned/' folder. Make sure files are named like SYMBOL_cleaned.csv")
    exit(1)

for file_path in cleaned_files:
    filename = os.path.basename(file_path)
    symbol = filename.replace("_cleaned.csv", "")
    print(f"\nProcessing {symbol} ...")

    # load
    df = pd.read_csv(file_path)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # ensure Close exists and numeric
    if "Close" not in df.columns:
        print(f"  SKIP {symbol}: 'Close' column missing.")
        continue
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    # indicators
    df[f"SMA{SMA_SHORT}"] = sma(df["Close"], SMA_SHORT)
    df[f"SMA{SMA_LONG}"] = sma(df["Close"], SMA_LONG)
    df["RSI"] = rsi(df["Close"], RSI_PERIOD)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["Daily_Return"] = df["Close"].pct_change()

    # signals (1 buy, -1 sell, 0 neutral)
    # SMA crossover
    df["SMA_signal"] = 0
    df.loc[df[f"SMA{SMA_SHORT}"] > df[f"SMA{SMA_LONG}"], "SMA_signal"] = 1
    df.loc[df[f"SMA{SMA_SHORT}"] < df[f"SMA{SMA_LONG}"], "SMA_signal"] = -1
    df["SMA_cross"] = df["SMA_signal"].diff().fillna(0)  # 2->1 etc.

    # RSI signal
    df["RSI_signal"] = 0
    df.loc[df["RSI"] < 30, "RSI_signal"] = 1     # oversold -> buy
    df.loc[df["RSI"] > 70, "RSI_signal"] = -1    # overbought -> sell

    # MACD crossover signal
    df["MACD_signal_flag"] = 0
    df.loc[df["MACD"] > df["MACD_signal"], "MACD_signal_flag"] = 1
    df.loc[df["MACD"] < df["MACD_signal"], "MACD_signal_flag"] = -1

    # Combined signal (voting): sum of individual signals -> final_signal: 1 (buy) if sum>0, -1 if sum<0, else 0
    df["signal_vote"] = df["SMA_signal"].fillna(0) + df["RSI_signal"].fillna(0) + df["MACD_signal_flag"].fillna(0)
    df["final_signal"] = 0
    df.loc[df["signal_vote"] > 0, "final_signal"] = 1
    df.loc[df["signal_vote"] < 0, "final_signal"] = -1

    # Simple backtest: enter long when final_signal==1, exit when final_signal==-1 (position 1 or 0)
    df["position"] = 0
    position = 0
    for i in range(len(df)):
        sig = df.at[i, "final_signal"]
        if sig == 1 and position == 0:
            position = 1
        elif sig == -1 and position == 1:
            position = 0
        df.at[i, "position"] = position

    # Strategy returns: when position=1, strategy_return = daily_return, else 0
    df["strategy_return"] = df["position"].shift(1).fillna(0) * df["Daily_Return"]
    df["cum_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["cum_buy_hold"] = (1 + df["Daily_Return"].fillna(0)).cumprod()

    # Metrics
    total_return_strategy = df["cum_strategy"].iloc[-1] - 1
    total_return_bh = df["cum_buy_hold"].iloc[-1] - 1
    # simple annualized approx: (1+ret)^(252/len)* -1
    days = len(df)
    ann_strategy = (1 + total_return_strategy) ** (252 / max(days, 1)) - 1 if days>1 else np.nan
    ann_bh = (1 + total_return_bh) ** (252 / max(days, 1)) - 1 if days>1 else np.nan
    sharpe = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(252) if df["strategy_return"].std() != 0 else np.nan

    metrics.append({
        "symbol": symbol,
        "rows": days,
        "total_return_strategy": total_return_strategy,
        "total_return_buy_hold": total_return_bh,
        "ann_return_strategy": ann_strategy,
        "ann_return_buy_hold": ann_bh,
        "sharpe_strategy": sharpe
    })

    # save enriched csv
    out_csv = os.path.join(OUTPUT_FOLDER, f"{symbol}_signals.csv")
    df.to_csv(out_csv, index=False)

    # Plot combined strategy chart (price + SMA + buy/sell markers) and bottom panels MACD & RSI
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14,10), gridspec_kw={"height_ratios": [3,1,1]})
    ax = axes[0]
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    ax.plot(df["Date"], df[f"SMA{SMA_SHORT}"], label=f"SMA{SMA_SHORT}")
    ax.plot(df["Date"], df[f"SMA{SMA_LONG}"], label=f"SMA{SMA_LONG}")

    # buy points: where position changes from 0->1
    buys = df[(df["position"].diff() == 1) & (df["position"] == 1)]
    sells = df[(df["position"].diff() == -1) & (df["position"] == 0)]
    ax.scatter(buys["Date"], buys["Close"], marker="^", color="green", s=100, label="Buy")
    ax.scatter(sells["Date"], sells["Close"], marker="v", color="red", s=100, label="Sell")

    ax.set_title(f"{symbol} Price, SMA & Strategy Signals")
    ax.legend()
    ax.grid(True)

    # MACD panel
    ax2 = axes[1]
    ax2.plot(df["Date"], df["MACD"], label="MACD")
    ax2.plot(df["Date"], df["MACD_signal"], label="MACD Signal")
    ax2.fill_between(df["Date"], df["MACD_hist"], 0, alpha=0.3)
    ax2.legend()
    ax2.set_ylabel("MACD")

    # RSI panel
    ax3 = axes[2]
    ax3.plot(df["Date"], df["RSI"], label="RSI", color="purple")
    ax3.axhline(70, color="red", linestyle="--")
    ax3.axhline(30, color="green", linestyle="--")
    ax3.set_ylabel("RSI")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_FOLDER, f"{symbol}_strategy.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved: {out_csv} and {out_png}")

# Save metrics summary
metrics_df = pd.DataFrame(metrics)
metrics_csv = os.path.join(OUTPUT_FOLDER, "strategy_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print("\nAll done. Metrics saved to:", metrics_csv)
