"""
Tests unitaires pour l'API de prédiction.
Exécution : pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# ----------------------------------------------------------------------
# Payload valide (exemple)
# ----------------------------------------------------------------------
VALID_PAYLOAD = {
    "Hours_Studied": 23,
    "Attendance": 84,
    "Parental_Involvement": "Low",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "No",
    "Sleep_Hours": 7,
    "Previous_Scores": 73,
    "Motivation_Level": "Low",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 0,
    "Family_Income": "Low",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "High School",
    "Distance_from_Home": "Near",
    "Gender": "Male"
}

# ----------------------------------------------------------------------
# Tests de santé
# ----------------------------------------------------------------------
def test_health_check():
    """Vérifie que l'endpoint /health retourne un statut ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# ----------------------------------------------------------------------
# Tests de prédiction avec payload valide
# ----------------------------------------------------------------------
def test_predict_valid_payload():
    """Un payload complet et correct doit retourner 200 et un score prédit."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "Exam_Score_predicted" in data
    assert isinstance(data["Exam_Score_predicted"], float)
    # La prédiction doit être dans une plage raisonnable (50-105)
    assert 50 <= data["Exam_Score_predicted"] <= 105

# ----------------------------------------------------------------------
# Tests de validation (payloads invalides)
# ----------------------------------------------------------------------
def test_predict_missing_field():
    """Un champ manquant doit retourner 422 (erreur de validation)."""
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["Attendance"]
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_invalid_category():
    """Une valeur catégorielle non autorisée doit retourner 422."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["Parental_Involvement"] = "Très Haut"  # N'existe pas
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_out_of_range_numeric():
    """Une valeur numérique hors limites (Attendance > 100) doit retourner 422."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["Attendance"] = 150  # max 100
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_negative_hours():
    """Une valeur négative pour Hours_Studied doit retourner 422."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["Hours_Studied"] = -5
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422