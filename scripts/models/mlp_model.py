import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim=1152):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.BatchNorm1d(input_dim // 8),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim // 8, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
