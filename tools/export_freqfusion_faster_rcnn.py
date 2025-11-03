#!/usr/bin/env python3
"""Exporta un paquete autocontenido con Faster R-CNN + FreqFusion."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "freqfusion_faster_rcnn_bundle"

# Archivos individuales a copiar (origen -> destino relativo dentro de EXPORT_ROOT)
FILE_MAP = {
    Path("FreqFusion.py"): Path("FreqFusion.py"),
    # Configuraciones
    Path("mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py"): Path("configs/_base_/models/faster_rcnn_r50_fpn.py"),
    Path("mmdetection/configs/_base_/datasets/coco_detection.py"): Path("configs/_base_/datasets/coco_detection.py"),
    Path("mmdetection/configs/_base_/schedules/schedule_1x.py"): Path("configs/_base_/schedules/schedule_1x.py"),
    Path("mmdetection/configs/_base_/default_runtime.py"): Path("configs/_base_/default_runtime.py"),
    Path("mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py"): Path("configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py"),
    Path("mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_freqfusion.py"): Path("configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_freqfusion.py"),
    # Scripts de entrenamiento/evaluación
    Path("mmdetection/tools/train.py"): Path("tools/train.py"),
    Path("mmdetection/tools/dist_train.sh"): Path("tools/dist_train.sh"),
    Path("mmdetection/tools/test.py"): Path("tools/test.py"),
    Path("mmdetection/tools/dist_test.sh"): Path("tools/dist_test.sh"),
    # Núcleo de FreqFusion dentro de MMDetection
    Path("mmdetection/mmdet/models/necks/FreqFusion.py"): Path("mmdet/models/necks/FreqFusion.py"),
    Path("mmdetection/mmdet/models/necks/fpn_carafe.py"): Path("mmdet/models/necks/fpn_carafe.py"),
    Path("mmdetection/mmdet/models/necks/fpn.py"): Path("mmdet/models/necks/fpn.py"),
    # Detectores y componentes de Faster R-CNN
    Path("mmdetection/mmdet/models/builder.py"): Path("mmdet/models/builder.py"),
    Path("mmdetection/mmdet/models/backbones/resnet.py"): Path("mmdet/models/backbones/resnet.py"),
    Path("mmdetection/mmdet/models/dense_heads/base_dense_head.py"): Path("mmdet/models/dense_heads/base_dense_head.py"),
    Path("mmdetection/mmdet/models/dense_heads/anchor_head.py"): Path("mmdet/models/dense_heads/anchor_head.py"),
    Path("mmdetection/mmdet/models/dense_heads/dense_test_mixins.py"): Path("mmdet/models/dense_heads/dense_test_mixins.py"),
    Path("mmdetection/mmdet/models/dense_heads/rpn_head.py"): Path("mmdet/models/dense_heads/rpn_head.py"),
    Path("mmdetection/mmdet/models/detectors/base.py"): Path("mmdet/models/detectors/base.py"),
    Path("mmdetection/mmdet/models/detectors/two_stage.py"): Path("mmdet/models/detectors/two_stage.py"),
    Path("mmdetection/mmdet/models/detectors/faster_rcnn.py"): Path("mmdet/models/detectors/faster_rcnn.py"),
    Path("mmdetection/mmdet/models/losses/accuracy.py"): Path("mmdet/models/losses/accuracy.py"),
    Path("mmdetection/mmdet/models/losses/cross_entropy_loss.py"): Path("mmdet/models/losses/cross_entropy_loss.py"),
    Path("mmdetection/mmdet/models/losses/smooth_l1_loss.py"): Path("mmdet/models/losses/smooth_l1_loss.py"),
    Path("mmdetection/mmdet/models/roi_heads/base_roi_head.py"): Path("mmdet/models/roi_heads/base_roi_head.py"),
    Path("mmdetection/mmdet/models/roi_heads/standard_roi_head.py"): Path("mmdet/models/roi_heads/standard_roi_head.py"),
    Path("mmdetection/mmdet/models/roi_heads/bbox_heads/bbox_head.py"): Path("mmdet/models/roi_heads/bbox_heads/bbox_head.py"),
    Path("mmdetection/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py"): Path("mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py"),
    Path("mmdetection/mmdet/models/roi_heads/roi_extractors/base_roi_extractor.py"): Path("mmdet/models/roi_heads/roi_extractors/base_roi_extractor.py"),
    Path("mmdetection/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py"): Path("mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py"),
    Path("mmdetection/mmdet/models/utils/res_layer.py"): Path("mmdet/models/utils/res_layer.py"),
    # API de alto nivel
    Path("mmdetection/mmdet/__init__.py"): Path("mmdet/__init__.py"),
    Path("mmdetection/mmdet/version.py"): Path("mmdet/version.py"),
    Path("mmdetection/mmdet/apis/__init__.py"): Path("mmdet/apis/__init__.py"),
    Path("mmdetection/mmdet/apis/train.py"): Path("mmdet/apis/train.py"),
    Path("mmdetection/mmdet/apis/test.py"): Path("mmdet/apis/test.py"),
    Path("mmdetection/mmdet/apis/inference.py"): Path("mmdet/apis/inference.py"),
    # Datasets
    Path("mmdetection/mmdet/datasets/builder.py"): Path("mmdet/datasets/builder.py"),
    Path("mmdetection/mmdet/datasets/coco.py"): Path("mmdet/datasets/coco.py"),
    Path("mmdetection/mmdet/datasets/custom.py"): Path("mmdet/datasets/custom.py"),
    Path("mmdetection/mmdet/datasets/dataset_wrappers.py"): Path("mmdet/datasets/dataset_wrappers.py"),
    Path("mmdetection/mmdet/datasets/utils.py"): Path("mmdet/datasets/utils.py"),
    Path("mmdetection/mmdet/datasets/pipelines/compose.py"): Path("mmdet/datasets/pipelines/compose.py"),
    Path("mmdetection/mmdet/datasets/pipelines/loading.py"): Path("mmdet/datasets/pipelines/loading.py"),
    Path("mmdetection/mmdet/datasets/pipelines/transforms.py"): Path("mmdet/datasets/pipelines/transforms.py"),
    Path("mmdetection/mmdet/datasets/pipelines/formatting.py"): Path("mmdet/datasets/pipelines/formatting.py"),
    Path("mmdetection/mmdet/datasets/pipelines/formating.py"): Path("mmdet/datasets/pipelines/formating.py"),
    Path("mmdetection/mmdet/datasets/pipelines/test_time_aug.py"): Path("mmdet/datasets/pipelines/test_time_aug.py"),
    Path("mmdetection/mmdet/datasets/api_wrappers/coco_api.py"): Path("mmdet/datasets/api_wrappers/coco_api.py"),
    Path("mmdetection/mmdet/datasets/samplers/distributed_sampler.py"): Path("mmdet/datasets/samplers/distributed_sampler.py"),
    Path("mmdetection/mmdet/datasets/samplers/group_sampler.py"): Path("mmdet/datasets/samplers/group_sampler.py"),
    # Utilidades generales
    Path("mmdetection/mmdet/utils/__init__.py"): Path("mmdet/utils/__init__.py"),
    Path("mmdetection/mmdet/utils/collect_env.py"): Path("mmdet/utils/collect_env.py"),
    Path("mmdetection/mmdet/utils/logger.py"): Path("mmdet/utils/logger.py"),
    Path("mmdetection/mmdet/utils/misc.py"): Path("mmdet/utils/misc.py"),
    Path("mmdetection/mmdet/utils/replace_cfg_vals.py"): Path("mmdet/utils/replace_cfg_vals.py"),
    Path("mmdetection/mmdet/utils/rfnext.py"): Path("mmdet/utils/rfnext.py"),
    Path("mmdetection/mmdet/utils/setup_env.py"): Path("mmdet/utils/setup_env.py"),
    Path("mmdetection/mmdet/utils/split_batch.py"): Path("mmdet/utils/split_batch.py"),
    Path("mmdetection/mmdet/utils/util_distribution.py"): Path("mmdet/utils/util_distribution.py"),
    Path("mmdetection/mmdet/utils/util_mixins.py"): Path("mmdet/utils/util_mixins.py"),
    Path("mmdetection/mmdet/utils/util_random.py"): Path("mmdet/utils/util_random.py"),
    Path("mmdetection/mmdet/utils/contextmanagers.py"): Path("mmdet/utils/contextmanagers.py"),
    Path("mmdetection/mmdet/utils/memory.py"): Path("mmdet/utils/memory.py"),
    Path("mmdetection/mmdet/utils/compat_config.py"): Path("mmdet/utils/compat_config.py"),
    Path("mmdetection/mmdet/utils/profiling.py"): Path("mmdet/utils/profiling.py"),
    Path("mmdetection/mmdet/utils/ascend_util.py"): Path("mmdet/utils/ascend_util.py"),
    # Documentación del bundle
    Path("docs/templates/freqfusion_faster_rcnn_bundle_README.md"): Path("README.md"),
}

# Directorios a copiar completamente (se mantienen todas sus rutas internas)
DIR_MAP = {
    Path("mmdetection/mmdet/core/anchor"): Path("mmdet/core/anchor"),
    Path("mmdetection/mmdet/core/bbox"): Path("mmdet/core/bbox"),
    Path("mmdetection/mmdet/core/data_structures"): Path("mmdet/core/data_structures"),
    Path("mmdetection/mmdet/core/evaluation"): Path("mmdet/core/evaluation"),
    Path("mmdetection/mmdet/core/post_processing"): Path("mmdet/core/post_processing"),
    Path("mmdetection/mmdet/core/utils"): Path("mmdet/core/utils"),
    Path("mmdetection/mmdet/core/visualization"): Path("mmdet/core/visualization"),
}

# Archivos de plantilla propios para sobreescribir los __init__ y dejar sólo las
# importaciones necesarias.
TEMPLATE_MAP = {
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/core/__init__.py"): Path("mmdet/core/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/datasets/__init__.py"): Path("mmdet/datasets/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/datasets/pipelines/__init__.py"): Path("mmdet/datasets/pipelines/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/datasets/samplers/__init__.py"): Path("mmdet/datasets/samplers/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/__init__.py"): Path("mmdet/models/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/backbones/__init__.py"): Path("mmdet/models/backbones/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/necks/__init__.py"): Path("mmdet/models/necks/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/dense_heads/__init__.py"): Path("mmdet/models/dense_heads/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/detectors/__init__.py"): Path("mmdet/models/detectors/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/losses/__init__.py"): Path("mmdet/models/losses/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/roi_heads/__init__.py"): Path("mmdet/models/roi_heads/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/roi_heads/bbox_heads/__init__.py"): Path("mmdet/models/roi_heads/bbox_heads/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/roi_heads/roi_extractors/__init__.py"): Path("mmdet/models/roi_heads/roi_extractors/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/utils/__init__.py"): Path("mmdet/models/utils/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/datasets/api_wrappers/__init__.py"): Path("mmdet/datasets/api_wrappers/__init__.py"),
    Path("docs/templates/freqfusion_faster_rcnn_bundle/mmdet/models/roi_heads/test_mixins.py"): Path("mmdet/models/roi_heads/test_mixins.py"),
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

    for src, dst in TEMPLATE_MAP.items():
        abs_src = ROOT / src
        if not abs_src.exists():
            raise FileNotFoundError(f"No se encontró la plantilla esperada: {abs_src}")
        abs_dst = EXPORT_ROOT / dst
        abs_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_src, abs_dst)

    total_items = len(FILE_MAP) + len(DIR_MAP) + len(TEMPLATE_MAP)
    print(f"Copiados {total_items} elementos a {EXPORT_ROOT}")


if __name__ == "__main__":
    export()
