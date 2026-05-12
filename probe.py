"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier for hallucination detection.

    Args:
        input_dim: Dimensionality of the input feature vector (from aggregation).
        hidden_dim: Size of the hidden layer (if using MLP). Default 128.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # выход — вероятность от 0 до 1
        )
        self._threshold = 0.5
        self._scaler = StandardScaler()
        self._is_fitted = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Probabilities of shape ``(batch_size, 1)`` in [0, 1].
        """
        return self.classifier(features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe classifier on features X and labels y.

        Args:
            X: Feature matrix of shape (n_samples, feature_dim)
            y: Labels of shape (n_samples,)

        Returns:
            self for method chaining
        """
        # Обучаем scaler
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        # Конвертируем в тензоры
        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y).float().view(-1, 1)

        # Настройки обучения
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        self.train()
        for epoch in range(50):  # 50 эпох
            optimizer.zero_grad()
            outputs = self(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

        self._is_fitted = True
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        self.eval()
        with torch.no_grad():
            prob_pos = self(X_t).numpy().flatten()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
