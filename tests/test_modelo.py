import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import joblib
import pytest

from src.model import FraudDetector
from src.utils import cargar_config


@pytest.fixture
def config():
    return cargar_config()


@pytest.fixture
def modelo(config):
    m = FraudDetector(30, config["hidden_layer"])
    m.load_state_dict(torch.load("models/fraud_model.pt",
                      map_location=torch.device("cpu")))
    m.eval()
    return m


@pytest.fixture
def scaler():
    return joblib.load("models/scaler.pkl")


def test_modelo_carga_correctamente(modelo):
    assert modelo is not None


def test_output_shape(modelo):
    X = torch.randn(1, 30)
    with torch.no_grad():
        output = modelo(X)
    assert output.shape == (1, 1)


def test_output_entre_0_y_1(modelo):
    X = torch.randn(10, 30)
    with torch.no_grad():
        output = modelo(X)
    assert output.min() >= 0
    assert output.max() <= 1


def test_gradientes(config):
    modelo = FraudDetector(30, config["hidden_layer"])
    X = torch.randn(8, 30)
    y = torch.randint(0, 2, (8, 1)).float()
    criterio = torch.nn.BCELoss()
    pred = modelo(X)
    loss = criterio(pred, y)
    loss.backward()
    assert modelo.red[0].weight.grad is not None


def test_scaler_tiene_30_features(scaler):
    assert scaler.n_features_in_ == 30