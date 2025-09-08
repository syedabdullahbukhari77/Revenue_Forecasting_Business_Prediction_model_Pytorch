# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.finance_model import finance_model
from training.preprocessing import load_and_preprocess
import joblib
import os

# 1. Load and preprocess data
X_train, X_val, y_train, y_val, scaler, encoders = load_and_preprocess("synthetic_financial_data_bukharii.csv")

# 2. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 3. Model
model = finance_model(input_dim=X_train.shape[1])
criterion = nn.MSELoss()   # basic loss, you can customize for multitask
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train loop (very simple for demo)
for epoch in range(20):  # increase epochs as needed
    model.train()
    optimizer.zero_grad()
    revenue, risk, churn = model(X_train)
    loss = criterion(revenue, y_train[:,0]) + criterion(risk, y_train[:,1]) + criterion(churn, y_train[:,2])
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# 5. Save artifacts
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/finance_model.pth")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Training complete. Model & preprocessing saved in /models/")
