import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py

DATA_DIR = "/home/matijens/esmc/data/embeddings"
OUTPUT_DIR = "data/new_features"
MODELS_DIR = "/home/matijens/esmc/models"
HIDDEN_DIM = 256
OUT_DIM = 64
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 15


class CreateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        optimizer: str = "AdamW",
        epochs: int = 50,
        patience: int = 8,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def _load_split(self, positive_file, negative_file):
        with h5py.File(positive_file, "r") as f:
            pos_emb = []
            for key in f.keys():
                pos_emb.append(f[key][:])

        with h5py.File(negative_file, "r") as f:
            neg_emb = []
            for key in f.keys():
                neg_emb.append(f[key][:])

        X = np.stack(pos_emb + neg_emb, axis=0).astype(np.float32)
        y = np.array([1] * len(pos_emb) + [0] * len(neg_emb))
        return X, y


device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = X_train.shape[1]

net = nn.Sequential(
    nn.Linear(input_dim, HIDDEN_DIM),
    nn.BatchNorm1d(HIDDEN_DIM),
    nn.ReLU(),
    nn.Dropout(DROPOUT),
    nn.Linear(HIDDEN_DIM, OUT_DIM),
)
head = nn.Linear(OUT_DIM, 2)
model = nn.Sequential(net, head).to(device)

dl = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
vX = torch.tensor(X_val, dtype=torch.float32).to(device)
vy = torch.tensor(y_val, dtype=torch.long).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

best_loss, best_state, wait = float("inf"), None, 0

for ep in range(EPOCHS):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        vl = loss_fn(model(vX), vy).item()

    if vl < best_loss:
        best_loss, best_state, wait = (
            vl,
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            0,
        )
    else:
        wait += 1
        if wait >= PATIENCE:
            break

model.load_state_dict(best_state)
model.eval()
net.cpu()

# === ZAPIS CHECKPOINTU MLP ===
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(
    {
        "state_dict": net.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "out_dim": OUT_DIM,
        "dropout": DROPOUT,
    },
    os.path.join(MODELS_DIR, "mlp.pt"),
)
print(f"Saved MLP checkpoint: {MODELS_DIR}/mlp.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split, pos_ids, neg_ids, X in [
    ("train", pos_train_ids, neg_train_ids, X_train),
    ("val", pos_val_ids, neg_val_ids, X_val),
    ("test", pos_test_ids, neg_test_ids, X_test),
]:
    n_pos = len(pos_ids)
    with torch.no_grad():
        features = net(torch.tensor(X, dtype=torch.float32)).numpy()

    pos_out = {sid: torch.tensor(features[i]) for i, sid in enumerate(pos_ids)}
    neg_out = {sid: torch.tensor(features[n_pos + i]) for i, sid in enumerate(neg_ids)}

    torch.save(pos_out, os.path.join(OUTPUT_DIR, f"positive_{split}.pt"))
    torch.save(neg_out, os.path.join(OUTPUT_DIR, f"negative_{split}.pt"))
