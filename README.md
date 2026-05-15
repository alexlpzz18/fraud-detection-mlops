# Fraud Detection MLOps

**Autor:** Alejandro López Lastra  
**Máster en Deep Learning — Universidad Politécnica de Madrid**  
**Asignatura:** MLOps

## Descripción

Sistema de detección de fraude bancario usando una red neuronal con PyTorch. 
Dado el historial de una transacción con tarjeta de crédito, el modelo predice 
si es fraudulenta o legítima.

El proyecto aplica las metodologías y herramientas de MLOps vistas en clase:
entorno reproducible, estructura profesional, tests automáticos, 
contenedorización con Docker y despliegue en producción.

## Dataset

Credit Card Fraud Detection — Kaggle  
284.807 transacciones reales, 492 fraudes (0.17% del total).

## Estructura del proyecto
fraud-detection-mlops/
├── config/
│   └── configuracion.yaml    # Hiperparámetros del modelo
├── src/
│   ├── model.py              # Arquitectura de la red neuronal
│   ├── train.py              # Script de entrenamiento
│   ├── utils.py              # Funciones auxiliares
│   └── api_inferencia.py     # API FastAPI
├── tests/
│   ├── test_datos.py         # Tests del dataset
│   ├── test_modelo.py        # Tests del modelo
│   └── test_api.py           # Tests de la API
├── .gitignore
├── requirements.txt
└── Dockerfile

## Requisitos

- Python 3.10
- Conda

## Instalación y uso local

**1 — Crear y activar el entorno:**
```bash
conda create -n fraud-detection python=3.10 -y
conda activate fraud-detection
pip install -r requirements.txt
```

**2 — Entrenar el modelo:**
```bash
python -m src.train
```

**3 — Lanzar la API en local:**
```bash
uvicorn src.api_inferencia:app --host 0.0.0.0 --port 8080
```

**4 — Ejecutar los tests:**
```bash
pytest tests/ -v
```

## Docker

**Construir la imagen:**
```bash
docker build -t fraud-detection:v1.0 .
```

**Ejecutar el contenedor:**
```bash
docker run -p 8080:8080 fraud-detection:v1.0
```

## API

La API expone dos endpoints:

**GET /** — Comprueba que el servidor está activo.

**POST /predice** — Recibe los datos de una transacción y devuelve la predicción.

Ejemplo de petición:
```json
{
  "Time": 0.0,
  "V1": -1.35,
  "Amount": 149.62
}
```

Ejemplo de respuesta:
```json
{
  "prediccion": "legítimo",
  "probabilidad_fraude": 0.0021
}
```

## Enlace W&B

[Ver experimentos en Weights & Biases](https://wandb.ai/alexlpzz-universidad-polit-cnica-de-madrid/fraud-detection)

## Endpoint en producción

Próximamente disponible tras el despliegue en Render.