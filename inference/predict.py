import torch
import numpy as np
from models.finance_model import finance_model

def load_model(path="models/finance_model.pth", input_dim=9, device="cpu"):
    model = finance_model(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict(model, features, device="cpu"):
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        revenue, risk, churn = model(tensor)

    return {
        "revenue": revenue.item(),
        "risk": risk.item(),
        "churn_probability": torch.sigmoid(churn).item()
    }
