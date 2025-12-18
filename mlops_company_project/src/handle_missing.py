import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "companies_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "companies_final.csv")

# Load data
df = pd.read_csv(INPUT_PATH)

# Handle missing values
df["Industry"] = df["Industry"].fillna("Unknown")
df["Description"] = df["Description"].fillna("Not Available")

# Fill Jobs with median
median_jobs = df["Jobs"].median()
df["Jobs"] = df["Jobs"].fillna(median_jobs)

# Save final dataset
df.to_csv(OUTPUT_PATH, index=False)

print("Missing values handled successfully")
