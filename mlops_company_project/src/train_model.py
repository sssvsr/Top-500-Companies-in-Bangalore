import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------
# Step 1: Load dataset
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "companies_cleaned_for_ml.csv")

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(df.head())

# -----------------------------
# Step 2: Features & Target
# -----------------------------
X = df.drop(columns=["Salaries"])
y = df["Salaries"]

# -----------------------------
# Step 3: Column types
# -----------------------------
numeric_features = ["Rating", "Reviews", "Jobs"]
categorical_features = ["Industry"]

# -----------------------------
# Step 4: Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# Step 5: Model Pipeline
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# -----------------------------
# Step 6: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 7: Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Step 8: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 9: Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# Step 10: Save Model
# -----------------------------
model_path = os.path.join(project_root, "models", "salary_model_with_industry.pkl")
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

joblib.dump(model, model_path)

print("\nModel with Industry saved successfully")
