# Python files relacionados con Faster R-CNN + FreqFusion

Esta guía reúne los archivos de Python que intervienen directamente en la integración de **FreqFusion** dentro de **Faster R-CNN** sobre COCO y los enlaces entre ellos. Úsala como mapa para revisar o portar la implementación.


> ✅ Encontrarás copias actualizadas de todos estos archivos dentro de la carpeta [`freqfusion_faster_rcnn_bundle/`](../freqfusion_faster_rcnn_bundle/), junto con **una copia completa del paquete `mmdet`**. Puedes quedarte únicamente con esa carpeta para trabajar con Faster R-CNN + FreqFusion.
=======
> ✅ Encontrarás copias actualizadas de todos estos archivos dentro de la carpeta [`freqfusion_faster_rcnn_bundle/`](../freqfusion_faster_rcnn_bundle/), lista para que puedas revisarlos o trasladarlos a otro proyecto sin depender de la estructura completa de este repositorio.


## Núcleo de FreqFusion

| Ruta | Descripción |
| --- | --- |
| `FreqFusion.py` | Implementación autocontenida del bloque FreqFusion usado como referencia y para experimentos fuera de MMDetection. |
| `mmdetection/mmdet/models/necks/FreqFusion.py` | Versión del bloque adaptada al ecosistema de MMDetection (usa registries, inicialización y utilidades del framework). |

## Cuello FPN con FreqFusion

| Ruta | Descripción |
| --- | --- |
| `mmdetection/mmdet/models/necks/fpn_carafe.py` | Define `FreqFusionCARAFEFPN`, que sustituye la interpolación tradicional del FPN por instancias de FreqFusion y se registra como cuello disponible. |
| `mmdetection/mmdet/models/necks/__init__.py` | Exporta los cuellos disponibles. Comprueba que el registro de `FreqFusionCARAFEFPN` esté accesible a través del registro `NECKS`. |

## Configuraciones de Faster R-CNN

| Ruta | Descripción |
| --- | --- |
| `mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py` | Configuración principal usada en el paper: reemplaza el FPN por `FreqFusionCARAFEFPN` y ajusta parámetros de entrenamiento. |
| `mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_freqfusion.py` | Variante con ResNet-101 como backbone. |

### Configuraciones base heredadas

El archivo anterior extiende los siguientes archivos de la carpeta `_base_`, por lo que forman parte del stack necesario para reproducir el experimento:

| Ruta | Descripción |
| --- | --- |
| `mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py` | Define el modelo Faster R-CNN con FPN estándar. Al reemplazar el `neck` por `FreqFusionCARAFEFPN`, se obtiene la variante con FreqFusion. |
| `mmdetection/configs/_base_/datasets/coco_detection.py` | Describe el dataset COCO, el pipeline de datos y los formatos de anotación. |
| `mmdetection/configs/_base_/schedules/schedule_1x.py` | Establece el plan de entrenamiento de 12 épocas (schedule 1x) usado en los experimentos. |
| `mmdetection/configs/_base_/default_runtime.py` | Ajustes de runtime compartidos (hooks, logger, checkpoints, etc.). |

## Scripts de entrenamiento y evaluación


## Implementación de Faster R-CNN en MMDetection

El detector de Faster R-CNN y sus componentes quedan empaquetados dentro de `freqfusion_faster_rcnn_bundle/mmdet/`. Los archivos más relevantes son:

| Ruta | Descripción |
| --- | --- |
| `mmdet/models/detectors/faster_rcnn.py` | Clase `FasterRCNN`, registrada en `DETECTORS`. |
| `mmdet/models/detectors/two_stage.py` | Lógica base para detectores de dos etapas (usada por Faster R-CNN). |
| `mmdet/models/roi_heads/standard_roi_head.py` | Define el ROI head estándar con el box head y máscara opcional. |
| `mmdet/models/rpn_heads/rpn_head.py` | Cabecera de la RPN. |
| `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py` | Cabeza de regresión/clasificación usada en COCO. |
| `mmdet/models/backbones/resnet.py` | Backbone ResNet con FPN. |
| `mmdet/core/` | Funciones auxiliares (asignadores, generadores de anchors, etc.). |

Al conservar la carpeta `freqfusion_faster_rcnn_bundle/` tendrás disponibles todas estas implementaciones sin depender del resto del repositorio.


| Ruta | Descripción |
| --- | --- |
| `mmdetection/tools/train.py` | Punto de entrada para entrenar Faster R-CNN + FreqFusion en una GPU. |
| `mmdetection/tools/dist_train.sh` | Script auxiliar para entrenamiento distribuido en múltiples GPUs. |
| `mmdetection/tools/test.py` | Evaluación del checkpoint entrenado sobre COCO. |
| `mmdetection/tools/dist_test.sh` | Evaluación distribuida. |

## Cómo actualizar el paquete de archivos

Ejecuta el script [`tools/export_freqfusion_faster_rcnn.py`](../tools/export_freqfusion_faster_rcnn.py) para copiar todos los archivos listados a `freqfusion_faster_rcnn_bundle/`. Esto te deja una carpeta autocontenida con la configuración y los módulos necesarios para estudiar o portar la implementación a otro proyecto.

```bash
python tools/export_freqfusion_faster_rcnn.py
```


> Nota: El paquete incluye el código completo de MMDetection v2.28.1 adaptado con FreqFusion. Únicamente necesitas instalar las dependencias de Python (MMCV, MMEngine, etc.) para ejecutarlo.

> Nota: La carpeta resultante no sustituye a MMDetection; solo agrupa los archivos clave. Para ejecutar Faster R-CNN + FreqFusion sigues necesitando las dependencias del framework original.

