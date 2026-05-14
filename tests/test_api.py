import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from src.api_inferencia import app

transaccion_valida = {
    "Time": 0.0,
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_raiz_responde(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "mensaje" in resp.json()


def test_predice_responde_200(client):
    resp = client.post("/predice", json=transaccion_valida)
    assert resp.status_code == 200


def test_predice_tiene_campos_correctos(client):
    resp = client.post("/predice", json=transaccion_valida)
    datos = resp.json()
    assert "prediccion" in datos
    assert "probabilidad_fraude" in datos


def test_prediccion_es_valida(client):
    resp = client.post("/predice", json=transaccion_valida)
    datos = resp.json()
    assert datos["prediccion"] in ["fraude", "legítimo"]
    assert 0 <= datos["probabilidad_fraude"] <= 1


def test_entrada_incorrecta_da_error(client):
    resp = client.post("/predice", json={"campo_inventado": 123})
    assert resp.status_code == 422