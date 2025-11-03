"""Wrappers mínimos sobre la API de COCO."""

from .coco_api import COCO, COCOeval  # noqa: F401

__all__ = ['COCO', 'COCOeval']
