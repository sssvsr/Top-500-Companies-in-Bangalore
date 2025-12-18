import pandas as pd
import os

# -----------------------------
# Step 1: Load dataset safely
# -----------------------------
# Get project root dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "companies_final.csv")

df = pd.read_csv(data_path)
print("First 5 rows of dataset:")
print(df.head())

# -----------------------------
# Step 2: Check datatypes
# -----------------------------
print("\nData types:")
print(df.dtypes)

# -----------------------------
# Step 3: Convert numeric columns
# -----------------------------
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
df["Salaries"] = pd.to_numeric(df["Salaries"], errors="coerce")
df["Jobs"] = pd.to_numeric(df["Jobs"], errors="coerce")

# -----------------------------
# Step 4: Basic stats
# -----------------------------
print("\nSummary statistics:")
print(df.describe())

# -----------------------------
# Step 5: Drop unnecessary columns
# -----------------------------
df_ml = df.drop(columns=["Description", "CompanyName", "Location"])

# -----------------------------
# Step 6: Save cleaned dataset
# -----------------------------
output_path = os.path.join(project_root, "data", "companies_cleaned_for_ml.csv")
df_ml.to_csv(output_path, index=False)
print("\nEDA and basic cleaning done. Dataset saved as companies_cleaned_for_ml.csv")
