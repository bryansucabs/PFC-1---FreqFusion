#!/usr/bin/env python3
"""Exporta los archivos clave de Faster R-CNN + FreqFusion a una carpeta de referencia."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "freqfusion_faster_rcnn_bundle"

# Lista de archivos a copiar: (origen, destino relativo dentro de EXPORT_ROOT)
FILE_MAP = {
    Path("FreqFusion.py"): Path("FreqFusion.py"),
    Path("mmdetection/mmdet/models/necks/FreqFusion.py"): Path("mmdet/models/necks/FreqFusion.py"),
    Path("mmdetection/mmdet/models/necks/fpn_carafe.py"): Path("mmdet/models/necks/fpn_carafe.py"),
    Path("mmdetection/mmdet/models/necks/__init__.py"): Path("mmdet/models/necks/__init__.py"),
    Path("mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py"): Path("configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py"),
    Path("mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_freqfusion.py"): Path("configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_freqfusion.py"),
    Path("mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py"): Path("configs/_base_/models/faster_rcnn_r50_fpn.py"),
    Path("mmdetection/configs/_base_/datasets/coco_detection.py"): Path("configs/_base_/datasets/coco_detection.py"),
    Path("mmdetection/configs/_base_/schedules/schedule_1x.py"): Path("configs/_base_/schedules/schedule_1x.py"),
    Path("mmdetection/configs/_base_/default_runtime.py"): Path("configs/_base_/default_runtime.py"),
    Path("mmdetection/tools/train.py"): Path("tools/train.py"),
    Path("mmdetection/tools/dist_train.sh"): Path("tools/dist_train.sh"),
    Path("mmdetection/tools/test.py"): Path("tools/test.py"),
    Path("mmdetection/tools/dist_test.sh"): Path("tools/dist_test.sh"),

    Path("docs/templates/freqfusion_faster_rcnn_bundle_README.md"): Path("README.md"),
}

DIR_MAP = {
    Path("mmdetection/mmdet"): Path("mmdet"),

}


def export() -> None:
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)

    for src_dir, dst_dir in DIR_MAP.items():
        abs_src_dir = ROOT / src_dir
        if not abs_src_dir.exists():
            raise FileNotFoundError(f"No se encontró la carpeta esperada: {abs_src_dir}")
        abs_dst_dir = EXPORT_ROOT / dst_dir
        shutil.copytree(abs_src_dir, abs_dst_dir, dirs_exist_ok=True)

    for src, dst in FILE_MAP.items():
        abs_src = ROOT / src
        if not abs_src.exists():
            raise FileNotFoundError(f"No se encontró el archivo esperado: {abs_src}")
        abs_dst = EXPORT_ROOT / dst
        abs_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_src, abs_dst)


    total_items = len(FILE_MAP) + len(DIR_MAP)
    print(f"Copiados {total_items} elementos a {EXPORT_ROOT}")
    print(f"Copiados {len(FILE_MAP)} archivos a {EXPORT_ROOT}")



if __name__ == "__main__":
    export()
