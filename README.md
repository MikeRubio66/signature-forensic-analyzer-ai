# Signature Analyzer AI

Modelo y API para análisis de firmas y detección de alteraciones básicas (demo). Incluye:

- pipeline de entrenamiento (CNN Keras)
- script de predicción
- API Flask para subir imagenes y obtener score
- Dockerfile y CI (GitHub Actions)

> ATTENTION: este proyecto es **demo educativo**. No debe usarse como único elemento probatorio en peritajes sin validación, documentación y pruebas exhaustivas.

## Requisitos
- Python 3.10+
- (Opcional) GPU para entrenamiento
- Datos de entrenamiento en `data/pos/` (firmas "válidas") y `data/neg/` (firmas "alteradas" o forgeries)

## Instalación
```bash
python -m venv venv
source venv/bin/activate   # linux/mac
venv\Scripts\activate      # windows
pip install -r requirements.txt
