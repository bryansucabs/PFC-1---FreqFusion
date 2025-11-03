"""Modelos reducidos para Faster R-CNN + FreqFusion."""

from .backbones import ResNet  # noqa: F401
from .builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS, ROI_EXTRACTORS,
                      SHARED_HEADS, build_backbone, build_detector, build_head,
                      build_loss, build_neck, build_roi_extractor,
                      build_shared_head)
from .dense_heads import AnchorHead, RPNHead  # noqa: F401
from .detectors import FasterRCNN  # noqa: F401
from .losses import Accuracy, accuracy, CrossEntropyLoss, L1Loss, SmoothL1Loss  # noqa: F401
from .necks import FPN, FPN_CARAFE, FreqFusionCARAFEFPN  # noqa: F401
from .roi_heads import StandardRoIHead  # noqa: F401
from .roi_heads.bbox_heads import BBoxHead, Shared2FCBBoxHead  # noqa: F401
from .roi_heads.roi_extractors import SingleRoIExtractor  # noqa: F401

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector'
]
