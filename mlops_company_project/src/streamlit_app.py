import streamlit as st
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Company Salary Predictor", page_icon="💼")

st.title("💼 Company Salary Prediction")
st.write("Enter company details to predict salary")

# -----------------------------
# Input fields
# -----------------------------
rating = st.number_input("Company Rating", min_value=1.0, max_value=5.0, value=4.2)
reviews = st.number_input("Number of Reviews", min_value=0, value=5000)
jobs = st.number_input("Number of Jobs", min_value=0, value=200)

industry = st.selectbox(
    "Industry",
    [
        "Business Consulting",
        "Accounting & Tax",
        "Enterprise Software & Network Solutions",
        "Information Technology Support Services",
        "Computer Hardware Development",
        "Stock Exchanges"
    ]
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Salary"):
    payload = {
        "Rating": rating,
        "Reviews": reviews,
        "Jobs": jobs,
        "Industry": industry
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"💰 Predicted Salary: {result['predicted_salary']:.2f}")
        else:
            st.error("API error. Please try again.")

    except Exception as e:
        st.error("FastAPI server is not running.")
