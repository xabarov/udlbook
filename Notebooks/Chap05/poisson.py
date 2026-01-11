# poisson_pedestrians_train.py
# Simulated data + PyTorch model that predicts pedestrian counts using a Poisson likelihood.
# Loss: Poisson negative log-likelihood (NLL) with log-rate output.

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# -------------------------
# Simulate dataset
# -------------------------
def simulate_data(N=50_000, K=8):
    """
    Features:
      - time of day: minutes in [0, 1439]  -> encoded with sin/cos
      - latitude/longitude: around a "city center"
      - neighborhood type: categorical in {0..K-1}
    Target:
      - y ~ Poisson(lambda(x))
    """
    # "City" center (arbitrary)
    lat0, lon0 = 55.75, 37.62

    # Sample features
    minutes = np.random.randint(0, 1440, size=N)  # 0..1439
    t = 2 * math.pi * (minutes / 1440.0)
    t_sin = np.sin(t)
    t_cos = np.cos(t)

    # Lat/lon: small area around center
    lat = lat0 + 0.03 * np.random.randn(N)
    lon = lon0 + 0.05 * np.random.randn(N)

    neigh = np.random.randint(0, K, size=N)

    # True generative model for log-rate
    # log(lambda) = bias + time effect + spatial effect + neighborhood effect + noise
    bias = 1.0

    # Time effect: morning/evening peaks via sin/cos combination
    w_sin, w_cos = 0.8, -0.4

    # Spatial effect: a linear trend (toy)
    w_lat, w_lon = 12.0, -7.0  # large-ish because lat/lon vary slightly

    # Neighborhood-specific offsets
    neigh_effect = np.linspace(-0.8, 0.9, K) + 0.2 * np.random.randn(K)

    # Small noise in log-space
    eps = 0.15 * np.random.randn(N)

    log_lambda = (
        bias
        + w_sin * t_sin
        + w_cos * t_cos
        + w_lat * (lat - lat0)
        + w_lon * (lon - lon0)
        + neigh_effect[neigh]
        + eps
    )

    lam = np.exp(log_lambda).astype(np.float32)
    y = np.random.poisson(lam).astype(np.int64)

    # Pack continuous features; we will standardize lat/lon for easier training
    X_cont = np.stack([t_sin, t_cos, lat, lon], axis=1).astype(np.float32)
    neigh = neigh.astype(np.int64)

    return X_cont, neigh, y


class PedCountDataset(Dataset):
    def __init__(self, X_cont, neigh, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.neigh = torch.tensor(neigh, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)  # PoissonNLLLoss expects float targets

    def __len__(self):
        return self.X_cont.shape[0]

    def __getitem__(self, idx):
        return self.X_cont[idx], self.neigh[idx], self.y[idx]


# -------------------------
# Model
# -------------------------
class PoissonReressor(nn.Module):
    """
    Predicts log-rate: log_lambda = f_theta(x), so lambda = exp(log_lambda) > 0
    """
    def __init__(self, num_neighborhoods, cont_dim=4, emb_dim=4, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(num_neighborhoods, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cont_dim + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # output log-lambda
        )

    def forward(self, x_cont, neigh):
        e = self.emb(neigh)
        x = torch.cat([x_cont, e], dim=1)
        log_lambda = self.mlp(x).squeeze(1)
        return log_lambda


# -------------------------
# Train / Eval
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Simulate
    K = 8
    X_cont, neigh, y = simulate_data(N=50_000, K=K)

    # Train/test split
    N = X_cont.shape[0]
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr, te = idx[:n_train], idx[n_train:]

    Xtr, ntr, ytr = X_cont[tr], neigh[tr], y[tr]
    Xte, nte, yte = X_cont[te], neigh[te], y[te]

    # Standardize lat/lon only (columns 2 and 3); keep sin/cos as-is
    latlon_mean = Xtr[:, 2:4].mean(axis=0, keepdims=True)
    latlon_std = Xtr[:, 2:4].std(axis=0, keepdims=True) + 1e-6
    Xtr[:, 2:4] = (Xtr[:, 2:4] - latlon_mean) / latlon_std
    Xte[:, 2:4] = (Xte[:, 2:4] - latlon_mean) / latlon_std

    train_ds = PedCountDataset(Xtr, ntr, ytr)
    test_ds = PedCountDataset(Xte, nte, yte)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = PoissonReressor(num_neighborhoods=K, cont_dim=4, emb_dim=4, hidden=64).to(device)

    # Poisson negative log-likelihood.
    # We output log_lambda directly, so set log_input=True.
    criterion = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n = 0
        with torch.no_grad():
            for x_cont, neigh_b, y_b in test_loader:
                x_cont = x_cont.to(device)
                neigh_b = neigh_b.to(device)
                y_b = y_b.to(device)

                log_lambda = model(x_cont, neigh_b)
                loss = criterion(log_lambda, y_b)

                # Predict expected count: E[y|x] = lambda = exp(log_lambda)
                y_hat = torch.exp(log_lambda)
                mae = torch.mean(torch.abs(y_hat - y_b))

                bs = y_b.shape[0]
                total_loss += loss.item() * bs
                total_mae += mae.item() * bs
                n += bs

        return total_loss / n, total_mae / n

    # Training loop
    for epoch in range(1, 11):
        model.train()
        running = 0.0
        n_seen = 0

        for x_cont, neigh_b, y_b in train_loader:
            x_cont = x_cont.to(device)
            neigh_b = neigh_b.to(device)
            y_b = y_b.to(device)

            log_lambda = model(x_cont, neigh_b)
            loss = criterion(log_lambda, y_b)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            bs = y_b.shape[0]
            running += loss.item() * bs
            n_seen += bs

        test_nll, test_mae = evaluate()
        print(
            f"epoch {epoch:02d} | train_nll={running/n_seen:.4f} | "
            f"test_nll={test_nll:.4f} | test_MAE(E[y])={test_mae:.4f}"
        )

    # Example predictions
    model.eval()
    x_cont, neigh_b, y_b = next(iter(test_loader))
    x_cont = x_cont.to(device)
    neigh_b = neigh_b.to(device)
    with torch.no_grad():
        log_lambda = model(x_cont, neigh_b)
        lam = torch.exp(log_lambda).cpu().numpy()

    print("\nSample predictions (lambda vs y):")
    for i in range(10):
        print(f"  lambda={lam[i]:7.3f}  y={int(y_b[i].item())}")

if __name__ == "__main__":
    main()
