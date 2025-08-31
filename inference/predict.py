# inference/predict.py
import torch
import numpy as np
from models.finance_model import finance_model


def load_model(model_path="models/finance_model.pth", input_dim=9, device="cpu"):
    """
    Load a trained finance_model from disk.
    """
    model = finance_model(input_dim=input_dim)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, features, device="cpu"):
    """
    Run inference on a single set of features.
    Args:
        model: Trained finance_model instance
        features: List or array of numeric inputs
        device: Torch device string ("cpu" or "cuda")
    Returns:
        dict: predictions for revenue, risk, churn_probability
    """
    features = np.asarray(features, dtype=np.float32).reshape(1, -1)
    features_tensor = torch.from_numpy(features).to(device)

    with torch.no_grad():
        revenue, risk, churn = model(features_tensor)

    return {
        "revenue": float(revenue.item()),
        "risk": float(risk.item()),
        "churn_probability": float(torch.sigmoid(churn).item()),
    }
