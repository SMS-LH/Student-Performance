"""Chargement du modèle final et prédiction."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_pipeline.pkl"

def load_model() -> Pipeline:
    """Charge le pipeline sauvegardé."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict(input_data) -> float:
    """
    Prend un dict ou un DataFrame d'un seul étudiant.
    Retourne la prédiction du score.
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame) and len(input_data) != 1:
        raise ValueError("Un seul échantillon à la fois.")
    pipeline = load_model()
    return float(pipeline.predict(input_data)[0])