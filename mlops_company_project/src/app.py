from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Company Salary Prediction API")

# -----------------------------
# Load model
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "salary_model_with_industry.pkl")

model = joblib.load(model_path)

# -----------------------------
# Input schema
# -----------------------------
class CompanyInput(BaseModel):
    Rating: float
    Reviews: int
    Jobs: int
    Industry: str

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_salary(data: CompanyInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_salary": prediction[0]}
