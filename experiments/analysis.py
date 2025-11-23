import pandas as pd
import os

print("\n=== STOCK ANALYSIS TOOL ===")

symbol = input("Enter stock symbol (example: MM_NS): ").strip()

# Always use CLEANED folder
clean_file = f"cleaned/{symbol}_cleaned.csv"

if not os.path.exists(clean_file):
    print(f"\n❌ ERROR: Cleaned file not found: {clean_file}")
    print("Run clean_all_csvs.py first!")
    exit()

# Load cleaned CSV
df = pd.read_csv(clean_file)

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print("\n=== SAMPLE DATA ===")
print(df.head())

print("\n=== BASIC STATISTICS ===")
print(df.describe(include="all"))

print("\n=== LATEST MARKET DATA ===")
print(df.tail(1))