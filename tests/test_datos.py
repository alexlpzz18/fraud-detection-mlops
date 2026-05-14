import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from src.utils import RUTA_DATOS


@pytest.fixture
def df():
    return pd.read_csv(RUTA_DATOS)


def test_columnas_correctas(df):
    columnas_esperadas = set(
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    )
    assert set(df.columns) == columnas_esperadas


def test_sin_nans(df):
    assert df.isna().sum().sum() == 0


def test_rango_amount(df):
    assert df["Amount"].min() >= 0
    assert df["Amount"].max() < 100000


def test_clases_validas(df):
    assert set(df["Class"].unique()) == {0, 1}


def test_dataset_no_vacio(df):
    assert len(df) > 0