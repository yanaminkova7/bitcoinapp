"""LSTM forecaster: sequences of the last `seq_len` closing prices predict the close
(or return) `horizon` days ahead - direct multi-step, same convention as the other ML
models in this project.

Normalization stats (mean/std) are computed only from the data passed to `fit()`, which
is always a training-only slice when called from walk-forward validation or the app - so
there is no leakage from future data into the scaling. The train/validation split inside
`fit()` is chronological (the last `val_fraction` of sequences, in time order) - never
shuffled, per the project's ML rules. Shuffling the *order in which training minibatches
are fed to the optimizer* during SGD is standard practice and does not leak information:
each (input sequence, target) pair is already a fixed, correctly time-bounded example
before shuffling; what must never be shuffled is which examples land in train vs.
validation, and that split is fixed before any shuffling happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.forecasting.base import BaseForecaster


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]
        return self.head(last_layer_hidden).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 8,
        val_fraction: float = 0.15,
        horizon: int = 1,
        target_type: str = "price",
        random_state: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.horizon = horizon
        self.target_type = target_type
        self.random_state = random_state

        self._net: _LSTMNet | None = None
        self._mean: float | None = None
        self._std: float | None = None
        self._last_window: np.ndarray | None = None
        self._last_close: float | None = None
        self.train_history: list[float] = []
        self.val_history: list[float] = []

    def _build_sequences(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        n_seq = n - self.seq_len - self.horizon + 1
        if n_seq <= 0:
            raise ValueError(
                f"Need at least {self.seq_len + self.horizon} rows to build one sequence, got {n}"
            )
        X_raw = np.empty((n_seq, self.seq_len), dtype=np.float64)
        y_raw = np.empty(n_seq, dtype=np.float64)
        base_close = np.empty(n_seq, dtype=np.float64)
        for i in range(n_seq):
            X_raw[i] = close[i : i + self.seq_len]
            base_close[i] = close[i + self.seq_len - 1]
            y_raw[i] = close[i + self.seq_len + self.horizon - 1]
        return X_raw, y_raw, base_close

    def fit(self, df: pd.DataFrame) -> "LSTMForecaster":
        torch.manual_seed(self.random_state)
        close = df["close"].to_numpy(dtype=np.float64)

        self._mean = float(close.mean())
        self._std = float(close.std()) or 1.0

        X_raw, y_raw, base_close = self._build_sequences(close)
        X_scaled = (X_raw - self._mean) / self._std

        if self.target_type == "price":
            y_scaled = (y_raw - self._mean) / self._std
        elif self.target_type == "return":
            y_scaled = (y_raw - base_close) / base_close
        else:
            raise ValueError(f"Unknown target_type: {self.target_type!r}")

        n_seq = len(X_scaled)
        n_val = max(1, int(n_seq * self.val_fraction))
        n_train = n_seq - n_val
        if n_train < self.batch_size:
            raise ValueError(f"Not enough sequences ({n_seq}) for training after reserving {n_val} for validation")

        X_train = torch.tensor(X_scaled[:n_train], dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_scaled[:n_train], dtype=torch.float32)
        X_val = torch.tensor(X_scaled[n_train:], dtype=torch.float32).unsqueeze(-1)
        y_val = torch.tensor(y_scaled[n_train:], dtype=torch.float32)

        self._net = _LSTMNet(self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.train_history = []
        self.val_history = []
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        generator = torch.Generator().manual_seed(self.random_state)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)

        for _epoch in range(self.epochs):
            self._net.train()
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, generator=generator
            )
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = self._net(X_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            train_loss = epoch_loss / n_train

            self._net.eval()
            with torch.no_grad():
                val_predictions = self._net(X_val)
                val_loss = loss_fn(val_predictions, y_val).item()

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

        self._last_window = close[-self.seq_len :]
        self._last_close = float(close[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._net is None or self._last_window is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon != self.horizon:
            raise ValueError(
                f"This model was trained to forecast {self.horizon} step(s) ahead (direct "
                f"multi-step, not iterative), so it cannot predict horizon={horizon}."
            )

        scaled_window = (self._last_window - self._mean) / self._std
        x = torch.tensor(scaled_window, dtype=torch.float32).reshape(1, self.seq_len, 1)
        with torch.no_grad():
            raw_output = float(self._net(x).item())

        if self.target_type == "price":
            price_prediction = raw_output * self._std + self._mean
        else:
            price_prediction = self._last_close * (1 + raw_output)

        return pd.Series([price_prediction])
