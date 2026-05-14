import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def cargar_config(ruta="config/configuracion.yaml"):
    with open(ruta, "r") as f:
        return yaml.safe_load(f)

RUTA_DATOS = "data/creditcard.csv"

def cargar_datos(ruta=RUTA_DATOS):
    df = pd.read_csv(ruta)
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)
    return X, y


def preprocesar(X, y, semilla=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=semilla, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

def preprocesar_transaccion(datos, scaler):
    datos_scaled = scaler.transform(datos)
    return datos_scaled