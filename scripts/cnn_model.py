import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, emb_dim=1152, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, hidden_dim, kernel_size=k),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            for k in (2, 3, 4)
        ])

        combined_dim = hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)

        pooled = [conv(x).max(dim=-1).values for conv in self.convs]
        x = torch.cat(pooled, dim=-1)  # (batch, hidden_dim * 3)

        return self.classifier(x).squeeze(-1)
