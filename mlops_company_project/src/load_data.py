import pandas as pd
import os

# Get absolute path to this file (src/load_data.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct path to data/companies.csv
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "companies.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# 1. Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# 2. Show dataset info
print("\nDataset Info:")
df.info()

# 3. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 4. Summary statistics
print("\nSummary statistics:")
print(df.describe())
