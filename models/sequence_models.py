"""
models/sequence_models.py
==========================
LSTM and GRU multi-step-ahead forecasters (PyTorch), plus shared
train/predict/evaluate utilities used identically by:
  - the RIME/MGO hyperparameter search (Phase 1 fitness function), and
  - the final "champion" model retraining and the GRU/LSTM baselines
    used in the forecasting benchmark table.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import get_torch_device
from utils.metrics import rmse, mape


def _device() -> torch.device:
    dev = get_torch_device()
    if dev is None:
        return torch.device("cpu")
    return dev


@dataclass
class SequenceModelConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    horizon: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 40


class LSTMForecaster(nn.Module):
    def __init__(self, cfg: SequenceModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                 # (batch, hidden_size)
        return self.head(self.head_dropout(last_hidden))


class GRUForecaster(nn.Module):
    """Benchmark model #1 (Phase-1 comparison): plain GRU, same head."""
    def __init__(self, cfg: SequenceModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon)

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.head(self.head_dropout(last_hidden))


def build_model(model_type: str, cfg: SequenceModelConfig) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMForecaster(cfg).to(_device())
    elif model_type == "gru":
        return GRUForecaster(cfg).to(_device())
    raise ValueError(f"Unknown model_type: {model_type}")


def train_forecaster(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      cfg: SequenceModelConfig, verbose: bool = False,
                      early_stopping_patience: Optional[int] = 8) -> dict:
    """Standard supervised training loop with MSE loss, Adam optimizer, and
    optional early stopping on validation RMSE (restores best weights)."""
    dev = _device()
    use_cuda = dev.type == "cuda"
    model = model.to(dev)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False,
        pin_memory=use_cuda,
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=use_cuda)
            yb = yb.to(dev, non_blocking=use_cuda)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_rmse = torch.sqrt(torch.mean((val_pred - y_val_t) ** 2)).item()
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % max(1, int(cfg.epochs) // 10) == 0):
            print(f"  epoch {epoch:3d} | train_loss={history['train_loss'][-1]:.5f} | val_rmse={val_rmse:.5f}")

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_rmse"] = best_val_rmse
    return history


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    dev = _device()
    model.eval().to(dev)
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    return model(X_t).cpu().numpy()


def evaluate_forecaster(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = predict(model, X)
    return {
        "rmse": rmse(y, y_pred),
        "mape": mape(y, y_pred),
    }
