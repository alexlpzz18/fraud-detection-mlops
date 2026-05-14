import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from src.model import FraudDetector
from src.utils import cargar_config, preprocesar_transaccion


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

modelo = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo, scaler

    logger.info("Cargando modelo y scaler...")

    config = cargar_config()

    modelo = FraudDetector(30, config['hidden_layer'])
    modelo.load_state_dict(torch.load("models/fraud_model.pt",
                           map_location=torch.device("cpu")))
    modelo.eval()

    scaler = joblib.load("models/scaler.pkl")

    logger.info("Modelo y scaler cargados correctamente")

    yield

    logger.info("Apagando servidor...")


app = FastAPI(lifespan=lifespan)


class Transaccion(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def raiz():
    return {"mensaje": "API de detección de fraude bancario activa"}


@app.post("/predice")
def predice(transaccion: Transaccion):
    try:
        logger.info(f"Petición recibida: {transaccion}")

        datos = np.array(list(transaccion.model_dump().values()),
                        dtype=np.float32).reshape(1, -1)
        datos_scaled = preprocesar_transaccion(datos, scaler)
        tensor = torch.tensor(datos_scaled, dtype=torch.float32)

        with torch.no_grad():
            probabilidad = modelo(tensor).item()

        prediccion = "fraude" if probabilidad > 0.5 else "legítimo"

        logger.info(f"Predicción: {prediccion} (probabilidad: {probabilidad:.4f})")

        return {
            "prediccion": prediccion,
            "probabilidad_fraude": round(probabilidad, 4)
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail="Error al procesar la transacción")