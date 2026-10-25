"""Application Flask : API de segmentation (/predict) + interface d'analyse de profils (/, /batch)."""
import io
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from engine import Engine, ModelInputError
from features import OCCUPATION_MAP, RAW_FEATURES, WORKCLASS_MAP

BASE_DIR = Path(__file__).resolve().parent
NUMERIC_RULES = {"age": (0, 120), "education_num": (1, 16), "hours_per_week": (0, 168)}
CATEGORY_RULES = {"workclass": set(WORKCLASS_MAP), "occupation": set(OCCUPATION_MAP)}
MAX_BATCH = 1000       # lignes max pour /predict et /analyze
MAX_CSV_ROWS = 5000    # lignes max pour l'analyse en lot


def validate_row(row):
    """Validation de /predict (K-Means seul). Retourne (ligne propre, liste d'erreurs)."""
    if not isinstance(row, dict):
        return None, ["objet JSON attendu"]
    errors, clean = [], {}
    for field, (low, high) in NUMERIC_RULES.items():
        value = row.get(field)
        if value is None:
            errors.append(f"{field} : champ obligatoire")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{field} : nombre attendu")
        elif not low <= value <= high:
            errors.append(f"{field} : doit être entre {low} et {high}")
        else:
            clean[field] = value
    for field, allowed in CATEGORY_RULES.items():
        value = row.get(field)
        if value == "?":
            value = "Unknown"
        if value is None:
            errors.append(f"{field} : champ obligatoire (utilisez 'Unknown' si inconnu)")
        elif value not in allowed:
            errors.append(f"{field} : valeur '{value}' inconnue (voir GET /schema)")
        else:
            clean[field] = value
    return (None, errors) if errors else (clean, [])


def create_app(model_dir=None, config_dir=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5_000_000  # 5 Mo (CSV compris)
    app.json.ensure_ascii = False
    logging.basicConfig(level=logging.INFO)

    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", BASE_DIR / "models"))
    config_dir = Path(config_dir or os.environ.get("CONFIG_DIR", BASE_DIR / "config"))
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"Modèle introuvable dans '{model_dir}'. Lancez d'abord : python train.py")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_dir / metadata["model_file"])  # chargé UNE seule fois
    names = {int(k): v for k, v in metadata["cluster_names"].items()}
    engine = Engine(model_dir, config_dir)

    if metadata["sklearn_version"] != sklearn.__version__:
        app.logger.warning("scikit-learn %s ici, mais le modèle a été entraîné avec %s",
                           sklearn.__version__, metadata["sklearn_version"])
    if engine.missing:
        app.logger.warning("Modèles non chargés (fichier absent) : %s", engine.missing)

    @app.context_processor
    def inject_globals():
        return {"engine_missing": engine.missing}

    # ------------------------------------------------------------------ API : segmentation seule
    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": metadata["model_version"],
                "trained_at": metadata["trained_at"],
                "models_loaded": sorted(engine.models), "models_missing": engine.missing}

    @app.get("/schema")
    def schema():
        return {
            "fields": {
                **{f: {"type": "number", "min": lo, "max": hi} for f, (lo, hi) in NUMERIC_RULES.items()},
                **{f: {"type": "string", "values": sorted(v)} for f, v in CATEGORY_RULES.items()},
            },
            "segments": names,
        }

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return {"error": "Corps JSON invalide ou en-tête Content-Type: application/json manquant"}, 400
        rows = payload if isinstance(payload, list) else [payload]
        if not rows or len(rows) > MAX_BATCH:
            return {"error": f"Envoyez entre 1 et {MAX_BATCH} lignes"}, 400

        cleaned, problems = [], {}
        for i, row in enumerate(rows):
            clean, errors = validate_row(row)
            if errors:
                problems[i] = errors
            else:
                cleaned.append(clean)
        if problems:
            return {"error": "Validation échouée", "details": problems}, 422

        start = time.perf_counter()
        df = pd.DataFrame(cleaned, columns=RAW_FEATURES)
        labels = pipeline.predict(df)
        distances = pipeline.transform(df)[np.arange(len(df)), labels]
        elapsed_ms = (time.perf_counter() - start) * 1000
        app.logger.info("predict n=%d model=%s %.1f ms", len(df), metadata["model_version"], elapsed_ms)

        return jsonify({
            "model_version": metadata["model_version"],
            "predictions": [
                {"cluster": int(c), "segment": names[int(c)], "distance_au_centre": round(float(d), 3)}
                for c, d in zip(labels, distances)
            ],
        })

    # ------------------------------------------------------------------ API : fiche profil (tous les modèles)
    @app.get("/models")
    def models():
        return {"loaded": sorted(engine.models), "missing": engine.missing,
                "fields": engine.schema, "optional_fields": sorted(engine.estimable)}

    @app.post("/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if payload is None:
            return {"error": "Corps JSON invalide ou en-tête Content-Type: application/json manquant"}, 400
        rows = payload if isinstance(payload, list) else [payload]
        if not rows or len(rows) > MAX_BATCH:
            return {"error": f"Envoyez entre 1 et {MAX_BATCH} lignes"}, 400
        try:
            results, errors = engine.analyze(rows)
        except ModelInputError as exc:
            return {"error": str(exc)}, 422
        if not results:
            return {"error": "Validation échouée", "details": errors}, 422
        return {"results": results, "errors": errors}

    # ------------------------------------------------------------------ Interface web
    @app.route("/", methods=["GET", "POST"])
    def index():
        values, errors, result = {}, [], None
        if request.method == "POST":
            values = {f["name"]: (request.form.get(f["name"]) or "").strip() for f in engine.schema}
            try:
                results, row_errors = engine.analyze([{k: (v or None) for k, v in values.items()}])
                if row_errors:
                    errors = row_errors[0]
                else:
                    result = results[0]
            except ModelInputError as exc:
                errors = [str(exc)]
        return render_template("index.html", schema=engine.schema, optional=engine.estimable,
                               values=values, errors=errors, result=result)

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        context = {"segments": sorted(engine.segment_names.values()), "error": None, "table": None,
                   "counts": None, "total": 0, "skipped": 0, "selected": ""}
        if request.method == "GET":
            return render_template("batch.html", **context)

        file = request.files.get("csv")
        selected = request.form.get("segment", "")
        context["selected"] = selected
        if file is None or not file.filename:
            context["error"] = "Choisissez un fichier CSV."
            return render_template("batch.html", **context)
        try:
            df = pd.read_csv(io.BytesIO(file.read()), sep=None, engine="python",
                             encoding="utf-8-sig", skipinitialspace=True)
        except Exception:
            context["error"] = "Fichier illisible : un CSV avec une ligne d'en-tête est attendu."
            return render_template("batch.html", **context)
        if len(df) > MAX_CSV_ROWS:
            context["error"] = f"Trop de lignes ({len(df)}). Maximum : {MAX_CSV_ROWS}."
            return render_template("batch.html", **context)

        df = df.astype(object).where(df.notna(), None)
        try:
            results, row_errors = engine.analyze(df.to_dict("records"))
        except ModelInputError as exc:
            context["error"] = str(exc)
            return render_template("batch.html", **context)
        if not results:
            first = next(iter(row_errors.items()), (0, ["aucune ligne valide"]))
            context["error"] = f"Aucune ligne valide. Exemple (ligne {first[0] + 1}) : {'; '.join(first[1])}"
            return render_template("batch.html", **context)

        counts = pd.Series([r["segment"] for r in results]).value_counts().to_dict()
        if selected:
            results = [r for r in results if r["segment"] == selected]
        table = engine.flatten(results)

        if request.form.get("action") == "export":
            payload = table.to_csv(index=False, sep=";").encode("utf-8-sig")
            return Response(payload, mimetype="text/csv",
                            headers={"Content-Disposition": "attachment; filename=analyse.csv"})

        context.update(table=table.head(200).to_html(classes="table table-sm table-striped", index=False, border=0),
                       counts=counts, total=len(results), skipped=len(row_errors))
        return render_template("batch.html", **context)

    # ------------------------------------------------------------------ Erreurs
    @app.errorhandler(404)
    def not_found(_):
        return {"error": "Route introuvable"}, 404

    @app.errorhandler(405)
    def not_allowed(_):
        return {"error": "Méthode non autorisée"}, 405

    @app.errorhandler(413)
    def too_large(_):
        return {"error": "Requête trop volumineuse"}, 413

    @app.errorhandler(Exception)
    def unexpected(exc):
        if isinstance(exc, HTTPException):
            return {"error": exc.description}, exc.code
        app.logger.exception("Erreur inattendue")
        return {"error": "Erreur interne"}, 500

    return app
