import torch
import torch.nn as nn

class finance_model(nn.Module):   
    def __init__(self, input_dim: int):
        super(finance_model, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.revenue_head = nn.Linear(64, 1)
        self.risk_head = nn.Linear(64, 1)
        self.churn_head = nn.Linear(64, 1)

    def forward(self, X):
        X = self.shared(X)
        revenue = self.revenue_head(X)
        risk = self.risk_head(X)
        churn = self.churn_head(X)
        return revenue.squeeze(), risk.squeeze(), churn.squeeze()
