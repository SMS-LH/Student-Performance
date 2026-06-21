#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Téléchargement du dataset StudentPerformanceFactors.csv depuis Kaggle."""

import shutil
from pathlib import Path
import kagglehub

def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Téléchargement du dataset depuis Kaggle...")
    cache_path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
    print(f"Cache : {cache_path}")

    src = Path(cache_path) / "StudentPerformanceFactors.csv"
    dst = raw_dir / "StudentPerformanceFactors.csv"
    shutil.copy(src, dst)
    print(f"Fichier copié vers {dst}")

if __name__ == "__main__":
    main()