"""Validation-only early stopping, independent of PyTorch."""
from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    mode: str = 'min'
    min_epochs: int = 20
    patience: int = 5
    min_delta: float = 0.001
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self):
        if self.mode not in ('min', 'max'):
            raise ValueError('mode must be min or max')
        if self.min_epochs < 1 or self.patience < 1 or self.min_delta < 0:
            raise ValueError('Invalid early stopping settings')

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise ValueError('Validation monitor is not finite')
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == 'min'
            else value > self.best + self.min_delta
        )
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience
