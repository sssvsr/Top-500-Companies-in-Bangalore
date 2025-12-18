import joblib
import os
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "salary_model_with_industry.pkl")

model = joblib.load(model_path)
print("Model loaded successfully")

# -----------------------------
# Create sample input
# -----------------------------
sample_data = {
    "Rating": [4.2],
    "Reviews": [5000],
    "Jobs": [200],
    "Industry": ["Information Technology Support Services"]
}

sample_df = pd.DataFrame(sample_data)

# -----------------------------
# Prediction
# -----------------------------
predicted_salary = model.predict(sample_df)

print("\nPredicted Salary:", predicted_salary[0])
