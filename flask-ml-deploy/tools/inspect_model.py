"""Affiche les colonnes d'entrée et les classes d'un modèle sauvegardé.

Usage :  python tools/inspect_model.py models/mon_modele.joblib
"""
import sys

import joblib

model = joblib.load(sys.argv[1])
print("Type             :", type(model).__name__)
print("Colonnes d'entrée:", list(getattr(model, "feature_names_in_", ["(non disponibles)"])))
print("Classes          :", list(getattr(model, "classes_", ["(régression ou clustering)"])))
