"""Préparation des features, partagée entre l'entraînement (train.py) et l'API (app.py).

Reprend exactement les transformations du notebook :
  - education_num -> education_level (4 niveaux)
  - valeurs manquantes de workclass / occupation -> "Unknown"
  - regroupement de workclass et occupation en grandes familles
"""
import pandas as pd

# Champs bruts que l'API reçoit (et que le modèle utilise)
RAW_FEATURES = ["age", "education_level", "hours_per_week", "workclass", "occupation"]

# Colonnes après préparation
NUMERIC_COLS = ["age", "education_level", "hours_per_week"]
CATEGORICAL_COLS = ["workclass_group", "occupation_group"]

EDUCATION_BINS = [0, 3, 8, 12, 16]
EDUCATION_LABELS = [1, 2, 3, 4]

OCCUPATION_MAP = {
    "Exec-managerial": "Management-Professional",
    "Prof-specialty": "Management-Professional",
    "Adm-clerical": "Admin-Sales-Tech",
    "Sales": "Admin-Sales-Tech",
    "Tech-support": "Admin-Sales-Tech",
    "Craft-repair": "Manual-Technical",
    "Machine-op-inspct": "Manual-Technical",
    "Transport-moving": "Manual-Technical",
    "Handlers-cleaners": "Manual-Technical",
    "Farming-fishing": "Manual-Technical",
    "Other-service": "Services",
    "Priv-house-serv": "Services",
    "Protective-serv": "Services",
    "Armed-Forces": "Services",
    "Unknown": "Unknown",
    "None": "Unknown",
}

WORKCLASS_MAP = {
    "Private": "Private",
    "Self-emp-not-inc": "Self-employed",
    "Self-emp-inc": "Self-employed",
    "Federal-gov": "Government",
    "State-gov": "Government",
    "Local-gov": "Government",
    "Without-pay": "Other-unknown",
    "Never-worked": "Other-unknown",
    "Unknown": "Other-unknown",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les champs bruts en colonnes prêtes pour le ColumnTransformer."""
    out = df[RAW_FEATURES].copy()

    for col in ("workclass", "occupation"):
        missing = out[col].isna() | (out[col] == "?")
        out[col] = out[col].where(~missing, "Unknown")

    out["education_level"] = pd.cut(
        out["education_num"], bins=EDUCATION_BINS, labels=EDUCATION_LABELS
    ).astype(int)
    out["workclass_group"] = out["workclass"].map(WORKCLASS_MAP)
    out["occupation_group"] = out["occupation"].map(OCCUPATION_MAP)

    unknown = out[CATEGORICAL_COLS].isna().any(axis=1)
    if unknown.any():
        bad = out.loc[unknown, ["workclass", "occupation"]].drop_duplicates()
        raise ValueError(f"Modalités inconnues : {bad.to_dict(orient='records')}")

    return out[NUMERIC_COLS + CATEGORICAL_COLS]
