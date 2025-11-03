"""Cabezas de bounding boxes compatibles con Faster R-CNN."""

from .bbox_head import BBoxHead  # noqa: F401
from .convfc_bbox_head import Shared2FCBBoxHead  # noqa: F401

__all__ = ['BBoxHead', 'Shared2FCBBoxHead']
