"""Nettoyage des données brutes."""
import pandas as pd

def cap_exam_score(data: pd.DataFrame, max_score: int = 100) -> pd.DataFrame:
    """Plafonne Exam_Score à max_score."""
    df = data.copy()
    if 'Exam_Score' in df.columns:
        df.loc[df['Exam_Score'] > max_score, 'Exam_Score'] = max_score
    return df