"""
API de prédiction du score d'examen
------------------------------------
Endpoint principal : POST /predict
Le modèle utilisé est une régression linéaire simple, encapsulée
dans un pipeline (préprocesseur + estimateur) sauvegardé avec joblib.
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ----------------------------------------------------------------------
# Chargement du pipeline sauvegardé
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_pipeline.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Modèle introuvable à l'emplacement {MODEL_PATH}. "
        "Assurez-vous d'avoir exécuté le notebook 02 pour générer le modèle."
    )

# ----------------------------------------------------------------------
# Schéma des données d'entrée (Pydantic)
# ----------------------------------------------------------------------
class StudentData(BaseModel):
    Hours_Studied: float = Field(..., ge=0, description="Heures d'étude par semaine")
    Attendance: float = Field(..., ge=0, le=100, description="Taux d'assiduité (%)")
    Parental_Involvement: Literal["Low", "Medium", "High"]
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Extracurricular_Activities: Literal["No", "Yes"]
    Sleep_Hours: float = Field(..., ge=0, description="Heures de sommeil par nuit")
    Previous_Scores: float = Field(..., ge=0, description="Scores antérieurs")
    Motivation_Level: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["No", "Yes"]
    Tutoring_Sessions: float = Field(..., ge=0, description="Nombre de séances de tutorat")
    Family_Income: Literal["Low", "Medium", "High"]
    Teacher_Quality: Literal["Low", "Medium", "High"]
    School_Type: Literal["Public", "Private"]
    Peer_Influence: Literal["Positive", "Neutral", "Negative"]
    Physical_Activity: float = Field(..., ge=0, description="Heures d'activité physique par semaine")
    Learning_Disabilities: Literal["No", "Yes"]
    Parental_Education_Level: Literal["High School", "College", "Postgraduate"]
    Distance_from_Home: Literal["Near", "Moderate", "Far"]
    Gender: Literal["Male", "Female"]

# ----------------------------------------------------------------------
# Initialisation de l'application FastAPI
# ----------------------------------------------------------------------
app = FastAPI(
    title="Student Performance Predictor",
    description="API pour prédire le score d'examen d'un étudiant à partir de ses caractéristiques.",
    version="1.0.0",
)

# ----------------------------------------------------------------------
# Endpoint de santé
# ----------------------------------------------------------------------
@app.get("/health", tags=["Santé"])
def health_check():
    """Vérifier que l'API est opérationnelle."""
    return {"status": "ok"}

# ----------------------------------------------------------------------
# Endpoint de prédiction
# ----------------------------------------------------------------------
@app.post("/predict", tags=["Prédiction"])
def predict(student: StudentData):
    """
    Prédit le score d'examen d'un étudiant.

    - **student** : objet JSON contenant les 19 caractéristiques de l'étudiant.
    - Retourne le score estimé (float).
    """
    try:
        # Conversion en DataFrame avec l'ordre exact des colonnes utilisé à l'entraînement
        input_df = pd.DataFrame([student.model_dump()])
        # Les colonnes doivent être dans le même ordre que X_train original
        columns_order = [
            "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
            "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
            "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
            "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
            "Parental_Education_Level", "Distance_from_Home", "Gender"
        ]
        input_df = input_df[columns_order]

        # Prédiction
        prediction = pipeline.predict(input_df)[0]

        return {"Exam_Score_predicted": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")