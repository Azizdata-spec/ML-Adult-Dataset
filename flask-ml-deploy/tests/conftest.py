"""Fabrique de petits modèles de démonstration (données synthétiques) pour tester le moteur
sans dépendre de vos vrais modèles ni de adult.data."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import train
from features import OCCUPATION_MAP, RAW_FEATURES, WORKCLASS_MAP

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
NUMERIC = {"age", "education_num", "hours_per_week"}
MARITAL = ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"]


def synthetic_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    occupations = [o for o in OCCUPATION_MAP if o not in ("Unknown", "None")]
    df = pd.DataFrame({
        "age": rng.integers(17, 80, n),
        "education_num": rng.integers(1, 17, n),
        "workclass": rng.choice([w for w in WORKCLASS_MAP if w != "Unknown"], n),
        "occupation": rng.choice(occupations, n),
        "hours_per_week": rng.integers(10, 80, n),
        "marital_status": rng.choice(MARITAL, n),
    })
    score = df["education_num"] / 16 + df["hours_per_week"] / 80 + rng.normal(0, 0.3, n)
    df["income"] = np.where(score > 1.1, ">50K", "<=50K")
    df["has_gain"] = (rng.random(n) < 0.2 + 0.3 * (df["age"] > 45)).astype(int)
    df["get_bourse"] = (rng.random(n) < 0.3 + 0.3 * (df["age"] < 30)).astype(int)
    return df


def make_pipe(features, estimator):
    num = [c for c in features if c in NUMERIC]
    cat = [c for c in features if c not in NUMERIC]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


@pytest.fixture(scope="session")
def demo_models(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    df = synthetic_data()
    registry = json.loads((CONFIG_DIR / "registry.json").read_text(encoding="utf-8"))["models"]
    for spec in registry:
        if spec["role"] == "segmentation":
            pipe = train.build_pipeline().fit(df[RAW_FEATURES])
            metadata = {
                "model_version": "test", "model_file": spec["file"], "trained_at": "2026-01-01T00:00:00+00:00",
                "cluster_names": {str(k): v for k, v in train.CLUSTER_NAMES.items()},
                "sklearn_version": __import__("sklearn").__version__,
            }
            (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        else:
            model = (RandomForestRegressor(n_estimators=15, random_state=0) if spec["task"] == "regression"
                     else RandomForestClassifier(n_estimators=15, random_state=0))
            pipe = make_pipe(spec["features"], model).fit(df[spec["features"]], df[spec["target"]])
        joblib.dump(pipe, out / spec["file"])
    return out
