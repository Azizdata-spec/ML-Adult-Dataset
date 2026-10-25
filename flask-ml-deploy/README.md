# Analyse de profils : 6 modèles, une seule fiche

Chaîne d'analyse :

    Formulaire (profil, éventuellement incomplet)
      1. Estimateurs   : hours_per_week (régression), occupation (classification)  -> complètent les champs vides
      2. Segmentation  : K-Means                                                    -> segment
      3. Scores        : income, has_gain, get_bourse (classifications binaires)    -> probabilités
      4. Recommandation: règles segment x scores                                    -> offre / formation

## 1. Installation (PyCharm : terminal intégré, dans le dossier qui contient app.py)
    pip install -r requirements.txt

## 2. Mettre vos modèles dans models/
- K-Means : `python train.py --data data/adult.data` (crée segmentation_v1.joblib + metadata.json)
- Les 5 autres : sauvegardez chacun comme un **Pipeline complet** (préprocessing + modèle) avec
  `joblib.dump(pipeline, "models/<nom>.joblib", compress=3)`, entraîné sur un DataFrame de colonnes brutes.
- Pour connaître les colonnes d'entrée d'un modèle déjà sauvegardé :
  `python tools/inspect_model.py models/mon_modele.joblib`

## 3. Adapter les 3 fichiers de config/
| Fichier | Rôle |
|---|---|
| `registry.json` | pour chaque modèle : `file`, `features` (colonnes d'entrée), `target`, `positive_label` (classifieurs binaires) |
| `schema.json` | champs du formulaire : type, bornes, modalités (ajoutez ici toute colonne utilisée par vos modèles) |
| `recommendations.json` | règles segment x scores -> offre (la première règle qui correspond gagne) |

Règles à respecter :
- Un modèle dont le fichier est absent est simplement ignoré (l'interface l'indique).
- Les estimateurs sont exécutés dans l'ordre du registre ; leurs `features` ne doivent pas contenir leur propre cible.
- Un champ qu'un estimateur sait produire (ex. hours_per_week) peut rester vide dans le formulaire : il est alors estimé.
- `positive_label` doit être l'une des classes du modèle (voir `tools/inspect_model.py`).

## 4. Tests
    pytest        # fonctionne sans vos modèles : il fabrique des modèles de démonstration synthétiques

## 5. Lancer
    flask --app app run --port 8000                              # développement
    gunicorn -w 2 --preload -b 0.0.0.0:8000 wsgi:app             # Linux / WSL / Docker
    waitress-serve --port=8000 wsgi:app                          # Windows

- Interface : http://localhost:8000/ (fiche profil) et http://localhost:8000/batch (CSV, filtre par segment, export)
- API : `POST /analyze` (tous les modèles), `POST /predict` (K-Means seul), `GET /health`, `GET /models`, `GET /schema`

Exemple :

    Invoke-RestMethod -Method Post -Uri http://localhost:8000/analyze -ContentType "application/json" `
      -Body '{"age":35,"education_num":13,"workclass":"Private","marital_status":"Divorced"}'

## 6. Docker
    docker build -t analyse-profils .
    docker run --rm -p 8000:8000 analyse-profils
