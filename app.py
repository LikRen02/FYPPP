import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load trained model and transformer (assuming you saved them earlier)
xgb_model = XGBClassifier()
xgb_model.load_model('xgb_model.json')  # Load saved XGBoost model

def preprocess_input(data, transformer):
    """Preprocess user inputs using the trained transformer."""
    return transformer.transform(data)

def main():
    st.title("Fraud Detection Application")

    # Input details from the user
    st.header("Transaction Details")

    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
    transaction_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)
    product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Toys", "Others"])
    quantity = st.number_input("Quantity", min_value=1, step=1)
    device_used = st.selectbox("Device Used", ["Mobile", "Laptop", "Tablet", "Desktop"])
    is_address_match = st.selectbox("Is Address Match", ["Yes", "No"])

    # Derived features
    day_of_week = transaction_date.weekday()
    transaction_day = transaction_date.day
    transaction_month = transaction_date.month

    # Create a DataFrame for user input
    user_input = pd.DataFrame({
        "Transaction Amount": [transaction_amount],
        "Transaction Hour": [transaction_hour],
        "Product Category": [product_category],
        "Quantity": [quantity],
        "Device Used": [device_used],
        "Is Address Match": [1 if is_address_match == "Yes" else 0],
        "Transaction DOW": [day_of_week],
        "Transaction Day": [transaction_day],
        "Transaction Month": [transaction_month]
    })

    # Load and apply the transformer
    with open("transformer.pkl", "rb") as f:
        transformer = pickle.load(f)

    preprocessed_input = preprocess_input(user_input, transformer)

    # Prediction
    if st.button("Predict Fraud"):  # Button to make predictions
        prediction = xgb_model.predict(preprocessed_input)[0]

        if prediction == 1:
            st.error("⚠️ This transaction is predicted to be FRAUDULENT!")
        else:
            st.success("✅ This transaction is predicted to be SAFE.")

if __name__ == "__main__":
    main()