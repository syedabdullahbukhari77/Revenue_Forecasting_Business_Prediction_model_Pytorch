# training/train.py
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from models.finance_model import finance_model
from training.preprocessing import load_and_preprocess


def main():
    # 1. Load and preprocess data
    data_file = "synthetic_financial_data_bukharii.csv"
    X_train, X_val, y_train, y_val, scaler, encoders = load_and_preprocess(data_file)

    # 2. Convert to tensors
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)

    # 3. Initialize model, loss, optimizer
    model = finance_model(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training loop (basic demo version)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        revenue, risk, churn = model(X_train)
        loss = (
            criterion(revenue, y_train[:, 0])
            + criterion(risk, y_train[:, 1])
            + criterion(churn, y_train[:, 2])
        )

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}/{epochs}] Loss: {loss.item():.4f}")

    # 5. Save model and preprocessing artifacts
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/finance_model.pth")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    print("Training complete. Model & preprocessing saved to /models/")


if __name__ == "__main__":
    main()

