"""Pérdidas empleadas por Faster R-CNN."""

from .accuracy import Accuracy, accuracy  # noqa: F401
from .cross_entropy_loss import CrossEntropyLoss  # noqa: F401
from .smooth_l1_loss import L1Loss, SmoothL1Loss  # noqa: F401

__all__ = ['Accuracy', 'accuracy', 'CrossEntropyLoss', 'SmoothL1Loss', 'L1Loss']
