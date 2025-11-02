# Paquete autocontenido: Faster R-CNN + FreqFusion

Este directorio reúne **todos** los archivos de código y configuración necesarios para
entrenar o evaluar Faster R-CNN con el cuello `FreqFusionCARAFEFPN` sobre COCO. Con
este paquete puedes descartar el resto del repositorio si únicamente te interesa este
experimento.

## Contenido principal

- `mmdet/`: copia completa del paquete de MMDetection v2.28.1 con las extensiones de
  FreqFusion ya integradas.
- `configs/`: configuraciones listas para ResNet-50 y ResNet-101 sobre COCO (schedule
  1x).
- `tools/`: scripts de entrenamiento y evaluación (`train.py`, `test.py`, `dist_*`).
- `FreqFusion.py`: versión de referencia del bloque para experimentos fuera del
  framework.

## Requisitos

1. Python 3.8+.
2. Dependencias de MMDetection (consulta `mmdetection/requirements` o instala
   directamente desde `requirements/runtime.txt`).
3. MMCV compatible (`mmcv-full==1.5.3` recomendado por los autores).

Ejemplo de instalación rápida en un entorno limpio:

```bash
pip install -r mmdet/requirements/runtime.txt
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install -e .
```

El último comando instala el paquete `mmdet` desde esta carpeta. Si prefieres no
instalarlo en editable, añade `PYTHONPATH=$(pwd)` al ejecutar los scripts de `tools/`.

## Uso

```bash
# Entrenamiento en una GPU
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py

# Evaluación (reemplaza la ruta del checkpoint)
python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py \
    work_dirs/faster_rcnn_r50_fpn_1x_coco_freqfusion/latest.pth \
    --eval bbox
```

Los resultados se almacenarán en `work_dirs/` dentro de este mismo paquete. Puedes
borrar cualquier carpeta que no esté bajo `freqfusion_faster_rcnn_bundle/` si no la
necesitas para otros proyectos.

## Cómo regenerar el paquete

Si haces cambios sobre los archivos fuente originales, actualiza este paquete con:

```bash
python tools/export_freqfusion_faster_rcnn.py
```

El script reconstruye el contenido completo sobre `freqfusion_faster_rcnn_bundle/`.
