# Revenue_Forecasting_Pytorch_Business_Prediction_Model.
# 📊 Deep Learning Prototype: KPI Prediction from Financial Data

## 🔍 Overview

This project explores using **deep learning** to predict 3 critical business KPIs from synthetic but realistic financial/accounting tabular data:

- **Revenue Growth** (regression)
- **Risk Score** (regression)
- **Customer Churn** (classification)

Unlike traditional rule-based or Excel-driven forecasting, this model learns nonlinear patterns and handles noisy, high-dimensional data — enabling smarter KPI prediction across industries and regions.

---

## 🧠 Why Deep Learning?

✅ Captures complex, nonlinear relationships  
✅ Handles multi-task learning (predicting all 3 KPIs in parallel)  
✅ Learns from trends, seasonality, and categorical embeddings  
✅ Adapts better than fixed formulas in evolving markets

---

## 📦 Dataset

- **500 companies** × **68 months** = ~34,000 rows
- Simulated with realistic trends, margins, debt, and churn behavior
- Features include:  
  - `revenue`, `gross_profit`, `operating_margin`, `debt_ratio`, `log_revenue`
  - Categorical: `industry`, `region`, `company_size`
  - Time-based: `customer_tenure`, `date`

📁 File: `synthetic_financial_data_bukharii.csv`

---

## 🏗️ Model Architecture

- Built using **PyTorch**
- MLP with:
  - Batch Normalization
  - Dropout regularization
  - Multi-head output (3 KPI heads)
  - Categorical embeddings

---

## 🏋️ Training Summary

- Split: `2020–2024` for training, `2024–2025` for testing (time-aware split)
- Optimizer: Adam
- Losses:
  - Revenue Growth: MSE
  - Risk Score: MSE
  - Churn: BCEWithLogitsLoss

📉 **Loss Curves**:

| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1     | 1.1861     | 1.2510          |
| 5     | 0.7073     | 0.7825          |
| 10    | 0.6764     | 0.6891          |
| 15    | 0.5981     | 0.6107          |
| 20    | 0.6054     | 0.6232          |

![Loss Curve](assets/loss_plot.png)

---

## 🧪 Evaluation (on Unseen Test Data)

| Metric               | Value   |
|----------------------|---------|
| RMSE (Revenue Growth)| 0.0623  |
| RMSE (Risk Score)    | 0.0875  |
| Churn AUC            | 0.8432  |
| Churn Accuracy       | 0.7860  |

✅ Model generalizes well and beats naive baselines.

---

## 🆚 Baseline Comparison

| Model        | Revenue RMSE | Risk RMSE | Churn AUC |
|--------------|--------------|-----------|-----------|
| Linear Model | 0.1102       | 0.1301    | 0.6532    |
| Rule-based   | 0.1243       | 0.1432    | 0.6040    |
| **Our DL**   | **0.0623**   | **0.0875**| **0.8432**|

➡️ Deep learning model outperforms traditional methods across all KPIs.

---

## 🔎 Explainability (WIP)

- SHAP values planned for:
  - Top revenue drivers
  - Churn explainer per region/size
- Goal: Trust & interpretability for business stakeholders

---

## 🚀 Next Steps

- Add rolling features (e.g., 3-month average revenue)
- Improve model with Transformer-based architectures
- Deploy as a FastAPI/Streamlit dashboard
- Connect to real ERP/accounting systems

---

## 🛠️ Usage

```bash
# Train the model
python train_model.py

# Evaluate on test set
python evaluate.py
