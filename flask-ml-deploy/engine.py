"""Moteur d'analyse : enchaîne les modèles pour produire une fiche profil.

Chaîne :  estimateurs (ex. heures, profession)  ->  segmentation  ->  scores  ->  recommandation

Tout est piloté par trois fichiers de config/ :
  registry.json         les modèles (rôle, fichier, colonnes d'entrée, cible)
  schema.json           les champs du formulaire (type, bornes, modalités)
  recommendations.json  les règles segment x scores -> offre
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ModelInputError(ValueError):
    """Un modèle ne peut pas tourner avec les données fournies."""


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _native(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


class Engine:
    def __init__(self, model_dir, config_dir):
        self.model_dir = Path(model_dir)
        config_dir = Path(config_dir)
        self.registry = _read_json(config_dir / "registry.json")["models"]
        self.schema = _read_json(config_dir / "schema.json")["fields"]
        self.rules = _read_json(config_dir / "recommendations.json")

        self.models, self.missing = {}, []
        for spec in self.registry:
            path = self.model_dir / spec["file"]
            if path.exists():
                self.models[spec["name"]] = joblib.load(path)  # chargé UNE seule fois
            else:
                self.missing.append(spec["name"])

        self.segment_names = {}
        for spec in self.by_role("segmentation"):
            names_file = self.model_dir / spec.get("names_file", "")
            if names_file.is_file():
                names = _read_json(names_file).get("cluster_names", {})
                self.segment_names = {int(k): v for k, v in names.items()}

    # ---- accès -------------------------------------------------------------------
    def by_role(self, role):
        return [s for s in self.registry if s["role"] == role and s["name"] in self.models]

    @property
    def estimable(self):
        """Champs qu'un modèle sait estimer : ils peuvent rester vides dans le formulaire."""
        return {s["target"] for s in self.by_role("estimator")}

    # ---- validation --------------------------------------------------------------
    def validate_record(self, rec):
        errors, clean = [], {}
        for field in self.schema:
            name, raw = field["name"], rec.get(field["name"])
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                if name in self.estimable:
                    clean[name] = None
                else:
                    errors.append(f"{field['label']} : champ obligatoire")
                continue
            if field["type"] == "number":
                try:
                    if isinstance(raw, bool):
                        raise ValueError
                    value = float(raw)
                except (TypeError, ValueError):
                    errors.append(f"{field['label']} : nombre attendu")
                    continue
                if not field.get("min", -np.inf) <= value <= field.get("max", np.inf):
                    errors.append(f"{field['label']} : doit être entre {field.get('min')} et {field.get('max')}")
                    continue
                clean[name] = value
            else:
                value = str(raw).strip()
                if value not in field["choices"]:
                    errors.append(f"{field['label']} : valeur '{value}' inconnue")
                    continue
                clean[name] = value
        return clean, errors

    # ---- analyse -----------------------------------------------------------------
    def analyze(self, records):
        """records : liste de dict bruts. Retourne (résultats, erreurs par numéro de ligne)."""
        rows, index, errors = [], [], {}
        for i, rec in enumerate(records):
            clean, errs = self.validate_record(rec if isinstance(rec, dict) else {})
            if errs:
                errors[i] = errs
            else:
                rows.append(clean)
                index.append(i)
        results = []
        if rows:
            names = [f["name"] for f in self.schema]
            df = pd.DataFrame(rows, columns=names)
            for field in self.schema:
                if field["type"] == "number":
                    df[field["name"]] = pd.to_numeric(df[field["name"]], errors="coerce")
                else:
                    df[field["name"]] = df[field["name"]].astype(object)
            results = self._run(df)
            for result, i in zip(results, index):
                result["row"] = i
        return results, errors

    def _frame(self, df, spec):
        cols = spec["features"]
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise ModelInputError(f"Modèle '{spec['name']}' : colonnes absentes {absent} (ajoutez-les à config/schema.json)")
        X = df[cols]
        empty = [c for c in cols if X[c].isna().any()]
        if empty:
            raise ModelInputError(f"Modèle '{spec['name']}' : valeurs manquantes pour {empty}")
        return X

    def _run(self, df):
        df = df.copy()
        n = len(df)

        # 1. Estimateurs : complètent les champs laissés vides
        estimations = {}
        for spec in self.by_role("estimator"):
            target = spec["target"]
            pred = self.models[spec["name"]].predict(self._frame(df, spec))
            if spec["task"] == "regression":
                pred = np.round(np.asarray(pred, dtype=float), 1)
            else:
                pred = np.asarray(pred, dtype=object)
            declared = df[target] if target in df.columns else pd.Series([None] * n, index=df.index, dtype=object)
            was_missing = declared.isna().to_numpy()
            filled = declared.astype(object).to_numpy().copy()
            filled[was_missing] = pred[was_missing]
            df[target] = filled
            if spec["task"] == "regression":
                df[target] = pd.to_numeric(df[target])
            estimations[target] = {"label": spec["label"], "pred": pred, "estimated": was_missing}

        # 2. Segmentation
        labels = distances = None
        segmentation = self.by_role("segmentation")
        if segmentation:
            spec = segmentation[0]
            pipe, X = self.models[spec["name"]], self._frame(df, spec)
            labels = pipe.predict(X)
            distances = pipe.transform(X)[np.arange(n), labels]

        # 3. Scores (probabilité de la classe positive)
        scores = {}
        for spec in self.by_role("score"):
            pipe, X = self.models[spec["name"]], self._frame(df, spec)
            positive = str(spec["positive_label"])
            if hasattr(pipe, "predict_proba"):
                classes = [str(c) for c in pipe.classes_]
                if positive not in classes:
                    raise ModelInputError(
                        f"Modèle '{spec['name']}' : positive_label '{positive}' absent des classes {classes}")
                proba = pipe.predict_proba(X)[:, classes.index(positive)]
            else:
                proba = (pd.Series(pipe.predict(X)).astype(str) == positive).astype(float).to_numpy()
            scores[spec["name"]] = (spec, proba, proba >= spec.get("threshold", 0.5))

        # 4. Assemblage + recommandation
        results = []
        for i in range(n):
            segment = self.segment_names.get(int(labels[i]), f"Cluster {int(labels[i])}") if labels is not None else None
            flags = {name: bool(positive[i]) for name, (_, _, positive) in scores.items()}
            results.append({
                "inputs": {f["name"]: _native(df[f["name"]].iloc[i]) for f in self.schema},
                "estimations": {
                    target: {"label": e["label"], "value": _native(df[target].iloc[i]),
                             "estimated": bool(e["estimated"][i]), "predicted": _native(e["pred"][i])}
                    for target, e in estimations.items()
                },
                "cluster": int(labels[i]) if labels is not None else None,
                "segment": segment,
                "distance": round(float(distances[i]), 3) if distances is not None else None,
                "scores": {
                    name: {"label": spec["label"], "proba": round(float(proba[i]), 3), "positive": flags[name]}
                    for name, (spec, proba, _) in scores.items()
                },
                "recommendation": self._recommend(segment, flags),
            })
        return results

    def _recommend(self, segment, flags):
        for rule in self.rules["rules"]:
            if rule.get("segment") not in (None, segment):
                continue
            if all(flags.get(name) == expected for name, expected in rule.get("if", {}).items()):
                return rule["offer"]
        return self.rules["default"]["offer"]

    # ---- export ------------------------------------------------------------------
    @staticmethod
    def flatten(results):
        rows = []
        for r in results:
            row = {"ligne": r["row"] + 1, **r["inputs"], "segment": r["segment"],
                   "distance_au_centre": r["distance"]}
            for name, s in r["scores"].items():
                row[f"{name}_proba"] = s["proba"]
                row[f"{name}_positif"] = s["positive"]
            for target, e in r["estimations"].items():
                row[f"{target}_estime"] = e["estimated"]
            row["offre_recommandee"] = r["recommendation"]
            rows.append(row)
        return pd.DataFrame(rows)
