#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Point d’entrée principal : exécute le pipeline complet d’entraînement."""

import pandas as pd
from src.pipeline import run_training_pipeline

DATA_PATH = "data/raw/StudentPerformanceFactors.csv"

def main():
    print("Chargement des données...")
    data = pd.read_csv(DATA_PATH)
    print(f"Données chargées : {data.shape}")

    print("Lancement du pipeline...")
    metrics = run_training_pipeline(data)

    print("\n=== Métriques finales ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()