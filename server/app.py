import streamlit as st
import pandas as pd
import torch
import joblib
import sys, os

# allow importing parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.finance_model import finance_model

st.title("📊 Business Forecasting App")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    model = finance_model(input_dim=9)
    model.load_state_dict(torch.load("models/finance_model.pth", map_location="cpu"))
    model.eval()

    scaler = joblib.load("models/scaler.pkl")
    X = scaler.transform(df.select_dtypes(include=['number']))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        revenue, risk, churn = model(X_tensor)

    st.write("### Predictions")
    st.write("Revenue:", revenue.numpy())
    st.write("Risk:", risk.numpy())
    st.write("Churn probability:", torch.sigmoid(churn).numpy())
