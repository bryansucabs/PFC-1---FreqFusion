"""Cabeceras densas necesarias para la RPN."""

from .anchor_head import AnchorHead  # noqa: F401
from .rpn_head import RPNHead  # noqa: F401

__all__ = ['AnchorHead', 'RPNHead']
