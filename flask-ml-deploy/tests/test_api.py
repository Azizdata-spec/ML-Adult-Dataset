import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app import create_app
from features import RAW_FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "metadata.json").exists(), reason="Lancez d'abord : python train.py"
)

VALID = {"age": 35, "education_num": 13, "hours_per_week": 45,
         "workclass": "Private", "occupation": "Exec-managerial"}


@pytest.fixture(scope="module")
def client():
    return create_app(MODEL_DIR).test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_schema(client):
    assert "workclass" in client.get("/schema").get_json()["fields"]


def test_predict_single(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    pred = r.get_json()["predictions"][0]
    assert pred["cluster"] in (0, 1, 2) and pred["segment"]


def test_predict_batch(client):
    r = client.post("/predict", json=[VALID, VALID])
    assert len(r.get_json()["predictions"]) == 2


def test_matches_training_sample(client):
    """Le cluster prédit par l'API doit être celui enregistré à l'entraînement."""
    sample = json.loads((MODEL_DIR / "sample.json").read_text(encoding="utf-8"))
    r = client.post("/predict", json=sample["input"])
    assert r.get_json()["predictions"][0]["cluster"] == sample["expected_cluster"]


def test_api_equals_direct_model(client):
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(MODEL_DIR / meta["model_file"])
    direct = int(model.predict(pd.DataFrame([VALID], columns=RAW_FEATURES))[0])
    assert client.post("/predict", json=VALID).get_json()["predictions"][0]["cluster"] == direct


def test_empty_body(client):
    assert client.post("/predict").status_code == 400


@pytest.mark.parametrize("patch", [
    {"age": None}, {"age": "trente"}, {"age": 500}, {"education_num": 0},
    {"workclass": "Astronaute"}, {"occupation": None},
])
def test_invalid_input_returns_422(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_unknown_route(client):
    assert client.get("/nope").status_code == 404
