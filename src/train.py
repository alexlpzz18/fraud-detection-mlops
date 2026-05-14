import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import joblib


from src.utils import cargar_config, cargar_datos, preprocesar
from src.model import FraudDetector


def fijar_semillas(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def entrenar():
    config = cargar_config()

    fijar_semillas(config["semilla"])

    wandb.init(
        project="fraud-detection",
        config=config
    )

    X, y = cargar_datos()
    X_train, X_test, y_train, y_test, scaler = preprocesar(X, y, config["semilla"])

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = X_train.shape[1]
    modelo = FraudDetector(input_dim, config["hidden_layer"])

    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config["learning_rate"])

    for epoca in range(config["epocas"]):
        modelo.train()
        perdida_total = 0

        for X_batch, y_batch in loader:
            optimizador.zero_grad()
            predicciones = modelo(X_batch)
            perdida = criterio(predicciones, y_batch)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item()

        modelo.eval()
        with torch.no_grad():
            preds_test = modelo(X_test_t)
            perdida_test = criterio(preds_test, y_test_t).item()
            accuracy = ((preds_test > 0.5) == y_test_t).float().mean().item()

        wandb.log({
            "epoca": epoca + 1,
            "perdida_train": perdida_total / len(loader),
            "perdida_test": perdida_test,
            "accuracy": accuracy
        })

        print(f"Época {epoca+1}/{config['epocas']} - "
              f"Loss train: {perdida_total/len(loader):.4f} - "
              f"Loss test: {perdida_test:.4f} - "
              f"Accuracy: {accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(modelo.state_dict(), "models/fraud_model.pt")
    print("Modelo guardado en models/fraud_model.pt")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler guardado en models/scaler.pkl")

    wandb.finish()


if __name__ == "__main__":
    entrenar()