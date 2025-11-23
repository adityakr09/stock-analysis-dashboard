import os
import pandas as pd

RAW_DIR = "data"
CLEAN_DIR = "cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_file(path, save_path):
    try:
        df = pd.read_csv(path)

        # Normalize column names
        df.columns = [col.strip().title() for col in df.columns]

        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        
        # Fix missing columns
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # Keep only expected
        df = df[expected_cols]

        # Fix date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Convert numbers
        num_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove empty rows
        df = df.dropna(subset=["Date", "Close"])

        df.to_csv(save_path, index=False)
        print(f"Cleaned: {save_path} ({len(df)} rows)")

    except Exception as e:
        print(f"Error cleaning {path}: {e}")


# MAIN LOOP
for file in os.listdir(RAW_DIR):
    if file.endswith(".csv"):
        raw_path = os.path.join(RAW_DIR, file)
        save_path = os.path.join(CLEAN_DIR, file)
        clean_file(raw_path, save_path)
