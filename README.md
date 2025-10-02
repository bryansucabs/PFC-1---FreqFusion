
# PFC-1 — Integración preliminar de fusión por frecuencia en Faster R-CNN (avance)

**Estado:** avance técnico inicial. Se preparó estructura de proyecto, configuración mínima y un prototipo de módulo de fusión por frecuencia ("FreqFusion") para insertarlo tras FPN en un *pipeline* Two-Stage (Faster R-CNN). 

## Cómo ejecutar (avance simulado)
```bash
# 1) crear entorno 
```

## Estructura
```
configs/   -> config de integración (plantilla)
src/       -> prototipo del módulo FreqFusion (wrapper)
scripts/   -> scripts de entrenamiento/evaluación (simulados en este avance)
logs/      -> registros de ejecución
results/   -> métricas preliminares y muestras
patches/   -> diff compacto de integración
```

## Notas
- Es un avanze de la implementación
