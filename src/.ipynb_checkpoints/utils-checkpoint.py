"""Fonctions utilitaires : chemin racine du projet."""
from pathlib import Path

def get_project_root() -> Path:
    """Retourne le dossier racine school-student-performance/."""
    return Path(__file__).resolve().parent.parent