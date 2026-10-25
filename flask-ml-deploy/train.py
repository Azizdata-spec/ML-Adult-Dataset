"""Entraîne le K-Means du notebook et sauvegarde le pipeline complet.

Usage :  python train.py --data data/adult.data --out models
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import CATEGORICAL_COLS, NUMERIC_COLS, RAW_FEATURES, prepare_features

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

MODEL_VERSION = "v1"
K_FINAL = 3
RANDOM_STATE = 42

# Noms choisis dans le notebook. Les numéros de cluster sont arbitraires :
# ils peuvent changer si les données ou la version de scikit-learn changent.
CLUSTER_NAMES = {
    0: "Seniors des métiers manuels et indépendants",
    1: "Cadres et professions intellectuelles",
    2: "Jeunes salariés du privé",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    return df.drop_duplicates()


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features)),
        ("preprocess", preprocessor),
        ("kmeans", KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)),
    ])


def check_cluster_names(profile: pd.DataFrame) -> None:
    """Vérifie que les numéros de cluster correspondent toujours aux noms du notebook."""
    id_of = {name: cid for cid, name in CLUSTER_NAMES.items()}
    cadres = id_of["Cadres et professions intellectuelles"]
    jeunes = id_of["Jeunes salariés du privé"]
    if profile["education_level"].idxmax() != cadres or profile["age"].idxmin() != jeunes:
        print("\nATTENTION : les numéros de cluster ne correspondent plus aux noms du notebook.")
        print(profile.round(2))
        print("Corrigez CLUSTER_NAMES dans train.py (ou utilisez --skip-name-check).")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/adult.data")
    parser.add_argument("--out", default="models")
    parser.add_argument("--skip-name-check", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)
    X_raw = df[RAW_FEATURES]
    print(f"{len(df)} lignes après suppression des doublons")

    pipe = build_pipeline().fit(X_raw)
    labels = pipe.predict(X_raw)

    X_t = pipe[:-1].transform(X_raw)
    silhouette = float(silhouette_score(X_t, labels, sample_size=10000, random_state=RANDOM_STATE))

    prepared = prepare_features(X_raw)
    profile = prepared.groupby(labels)[["age", "education_level", "hours_per_week"]].mean()
    profile["effectif_%"] = pd.Series(labels).value_counts(normalize=True).sort_index() * 100
    print("\nProfil des clusters :")
    print(profile.round(2))
    print(f"\nSilhouette (échantillon) : {silhouette:.3f}")

    if not args.skip_name_check:
        check_cluster_names(profile)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_file = f"segmentation_{MODEL_VERSION}.joblib"
    joblib.dump(pipe, out / model_file, compress=3)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_file": model_file,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "n_clusters": K_FINAL,
        "silhouette": round(silhouette, 4),
        "cluster_names": {str(k): v for k, v in CLUSTER_NAMES.items()},
        "features": RAW_FEATURES,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Exemple de référence : sert aux tests (le cluster attendu ne doit pas changer)
    sample = df.dropna(subset=RAW_FEATURES).iloc[[0]]
    sample_input = json.loads(sample[RAW_FEATURES].to_json(orient="records"))[0]
    expected = int(pipe.predict(sample[RAW_FEATURES])[0])
    (out / "sample.json").write_text(
        json.dumps({"input": sample_input, "expected_cluster": expected}, indent=2), encoding="utf-8"
    )
    print(f"\nModèle sauvegardé : {out / model_file}")


if __name__ == "__main__":
    main()
