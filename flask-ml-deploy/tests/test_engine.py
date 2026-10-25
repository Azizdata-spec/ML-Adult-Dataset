import io

import pytest

from app import create_app
from engine import Engine
from tests.conftest import CONFIG_DIR

FULL = {"age": 40, "education_num": 14, "workclass": "Private", "occupation": "Exec-managerial",
        "hours_per_week": 45, "marital_status": "Married-civ-spouse"}


@pytest.fixture(scope="module")
def engine(demo_models):
    return Engine(demo_models, CONFIG_DIR)


@pytest.fixture(scope="module")
def client(demo_models):
    return create_app(demo_models, CONFIG_DIR).test_client()


def test_all_models_loaded(engine):
    assert engine.missing == []
    assert len(engine.models) == 6


def test_full_profile(engine):
    results, errors = engine.analyze([FULL])
    r = results[0]
    assert not errors
    assert r["segment"] and r["recommendation"]
    assert set(r["scores"]) == {"income_clf", "has_gain_clf", "bourse_clf"}
    assert all(0 <= s["proba"] <= 1 for s in r["scores"].values())
    assert not r["estimations"]["hours_per_week"]["estimated"]  # valeur saisie conservée
    assert r["inputs"]["hours_per_week"] == 45


def test_missing_optional_fields_are_estimated(engine):
    results, _ = engine.analyze([{**FULL, "hours_per_week": None, "occupation": ""}])
    est = results[0]["estimations"]
    assert est["hours_per_week"]["estimated"] and est["occupation"]["estimated"]
    assert est["hours_per_week"]["value"] is not None
    assert results[0]["segment"]  # la segmentation utilise les valeurs estimées


def test_required_field_missing(engine):
    results, errors = engine.analyze([{**FULL, "age": None}])
    assert not results and 0 in errors


@pytest.mark.parametrize("patch", [{"workclass": "Astronaute"}, {"age": "abc"}, {"education_num": 99}])
def test_invalid_values(engine, patch):
    _, errors = engine.analyze([{**FULL, **patch}])
    assert 0 in errors


def test_batch_matches_single(engine):
    other = {**FULL, "age": 22, "education_num": 9, "occupation": "Sales"}
    batch, _ = engine.analyze([FULL, other, FULL])
    single, _ = engine.analyze([FULL])
    assert batch[0]["segment"] == batch[2]["segment"] == single[0]["segment"]
    assert batch[0]["scores"] == single[0]["scores"]


def test_invalid_rows_are_skipped_not_fatal(engine):
    results, errors = engine.analyze([FULL, {**FULL, "age": None}, FULL])
    assert len(results) == 2 and list(errors) == [1]
    assert [r["row"] for r in results] == [0, 2]


def test_recommendation_rules(engine):
    seg = "Cadres et professions intellectuelles"
    assert "leadership" in engine._recommend(seg, {"income_clf": True})
    assert "management" in engine._recommend(seg, {"income_clf": False})
    assert engine._recommend("Inconnu", {}) == engine.rules["default"]["offer"]


# ---- routes -------------------------------------------------------------------------
def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fiche profil" in r.get_data(as_text=True)


def test_form_submission(client):
    r = client.post("/", data=FULL)
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Recommandation" in html


def test_form_with_estimation(client):
    html = client.post("/", data={**FULL, "hours_per_week": "", "occupation": ""}).get_data(as_text=True)
    assert "estimé" in html


def test_form_error_shown(client):
    html = client.post("/", data={**FULL, "age": ""}).get_data(as_text=True)
    assert "champ obligatoire" in html


def test_analyze_json(client):
    r = client.post("/analyze", json=FULL)
    assert r.status_code == 200 and r.get_json()["results"][0]["recommendation"]
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze").status_code == 400


def test_models_route(client):
    body = client.get("/models").get_json()
    assert len(body["loaded"]) == 6 and "hours_per_week" in body["optional_fields"]


CSV = ("age,education_num,workclass,occupation,hours_per_week,marital_status\n"
       + "\n".join("40,14,Private,Exec-managerial,45,Married-civ-spouse" for _ in range(10))
       + "\n23,9,Private,Sales,,Never-married\n").encode()


def test_batch_page_and_export(client):
    assert client.get("/batch").status_code == 200
    data = {"csv": (io.BytesIO(CSV), "x.csv"), "action": "analyze", "segment": ""}
    r = client.post("/batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and "offre_recommandee" in r.get_data(as_text=True)
    data = {"csv": (io.BytesIO(CSV), "x.csv"), "action": "export", "segment": ""}
    r = client.post("/batch", data=data, content_type="multipart/form-data")
    assert r.mimetype == "text/csv" and "offre_recommandee" in r.get_data(as_text=True)


def test_batch_bad_file(client):
    data = {"csv": (io.BytesIO(b"a,b\n1,2\n"), "x.csv"), "action": "analyze", "segment": ""}
    r = client.post("/batch", data=data, content_type="multipart/form-data")
    assert "Aucune ligne valide" in r.get_data(as_text=True)
