import os
import pandas as pd

# Folders
RAW_FOLDER = "data"
CLEAN_FOLDER = "cleaned"

# Create cleaned folder if not exists
os.makedirs(CLEAN_FOLDER, exist_ok=True)

def clean_stock_csv(file_path, save_path):
    """
    Cleans a single stock CSV:
    - Removes junk rows
    - Fixes Date column
    - Converts numeric values
    """
    print(f"\nCleaning: {file_path}")

    # Read CSV with first row as header
    df = pd.read_csv(
        file_path,
        header=0,
        skiprows=[1, 2],      # remove Ticker row + Date row
        low_memory=False
    )

    # Fix missing Date column
    if "Date" not in df.columns:
        df.insert(0, "Date", pd.date_range(start="2000-01-01", periods=len(df)))

    # Convert Date column properly
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where Date failed parsing
    df = df.dropna(subset=["Date"])

    # Convert price columns to numeric
    for col in ["Close", "High", "Low", "Open", "Volume", "Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop empty rows
    df = df.dropna()

    # Save cleaned file
    df.to_csv(save_path, index=False)

    print(f"✔ Saved cleaned CSV → {save_path}")

# Process each CSV inside data/
for file in os.listdir(RAW_FOLDER):
    if file.endswith(".csv"):
        raw_path = os.path.join(RAW_FOLDER, file)
        clean_path = os.path.join(CLEAN_FOLDER, file)
        clean_stock_csv(raw_path, clean_path)

print("\n🎉 ALL CSV FILES CLEANED SUCCESSFULLY!")
