# analysis_all.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- CONFIG ----
CLEANED_FOLDER = "cleaned"    # folder where your cleaned CSVs are
OUTPUT_FOLDER = "outputs"     # outputs (CSVs + PNGs) will be saved here

STOCK_FILES = {
    "MM_NS": os.path.join(CLEANED_FOLDER, "MM_NS_cleaned.csv"),
    "SBIN_NS": os.path.join(CLEANED_FOLDER, "SBIN_NS_cleaned.csv"),
    "TITAN_NS": os.path.join(CLEANED_FOLDER, "TITAN_NS_cleaned.csv"),
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- helpers ----
def load_and_prepare(path):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure Date exists
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Convert numeric columns if present
    for col in ["Close", "Open", "High", "Low", "Volume", "Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ---- process each stock ----
dfs = {}
for sym, path in STOCK_FILES.items():
    if not os.path.exists(path):
        print(f"WARNING: file not found: {path}  — skipping {sym}")
        continue

    df = load_and_prepare(path)
    if "Close" not in df.columns:
        print(f"ERROR: 'Close' column missing in {path}. Skipping.")
        continue

    # Indicators
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility20"] = df["Daily_Return"].rolling(window=20, min_periods=1).std()

    # Save indicators CSV
    out_csv = os.path.join(OUTPUT_FOLDER, f"{sym}_indicators.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved indicators: {out_csv}")

    # Plot price + SMAs
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.plot(df["Date"], df["SMA20"], label="SMA20")
    ax.plot(df["Date"], df["SMA50"], label="SMA50")
    ax.set_title(f"{sym} - Close & SMAs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    out_png = os.path.join(OUTPUT_FOLDER, f"{sym}_price_sma.png")
    save_plot(fig, out_png)
    print(f"Saved plot: {out_png}")

    # Plot Volatility
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(df["Date"], df["Volatility20"], label="Volatility20")
    ax.set_title(f"{sym} - 20-day Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    out_png2 = os.path.join(OUTPUT_FOLDER, f"{sym}_volatility.png")
    save_plot(fig, out_png2)
    print(f"Saved volatility plot: {out_png2}")

    dfs[sym] = df

# ---- Combine close prices into one CSV for correlation ----
if not dfs:
    print("No valid stock data processed. Exiting.")
    exit()

# build a combined DataFrame on Date
combined = None
for sym, df in dfs.items():
    tmp = df[["Date", "Close"]].rename(columns={"Close": sym})
    if combined is None:
        combined = tmp
    else:
        combined = combined.merge(tmp, on="Date", how="outer")

combined = combined.sort_values("Date").reset_index(drop=True)
combined_csv = os.path.join(OUTPUT_FOLDER, "combined_closes.csv")
combined.to_csv(combined_csv, index=False)
print(f"Saved combined closes: {combined_csv}")

# correlation
corr = combined.set_index("Date").corr()
corr_csv = os.path.join(OUTPUT_FOLDER, "close_correlation.csv")
corr.to_csv(corr_csv)
print(f"Saved correlation matrix: {corr_csv}")

# heatmap
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.index)
ax.set_title("Correlation of Close Prices")
fig.colorbar(im, ax=ax)
heatmap_png = os.path.join(OUTPUT_FOLDER, "close_correlation.png")
save_plot(fig, heatmap_png)
print(f"Saved correlation heatmap: {heatmap_png}")

print("\nAll done. Check the 'outputs' folder for CSVs and PNGs.")
