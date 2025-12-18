import pandas as pd
import os
import numpy as np

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "companies.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "companies_cleaned.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Function to convert Indian numbers
def convert_indian_number(value):
    if pd.isna(value) or value == "--":
        return np.nan

    value = str(value).strip()

    if value.endswith("L"):
        return float(value.replace("L", "")) * 100000
    elif value.endswith("T"):
        return float(value.replace("T", "")) * 1000
    else:
        return float(value)

# Apply conversion
df["Reviews"] = df["Reviews"].apply(convert_indian_number)
df["Salaries"] = df["Salaries"].apply(convert_indian_number)
df["Jobs"] = df["Jobs"].apply(convert_indian_number)

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)

print("Data cleaned and saved successfully")