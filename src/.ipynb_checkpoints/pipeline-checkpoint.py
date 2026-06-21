"""Pipeline complet : raw → preprocessing → entraînement → sauvegarde."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SkPipeline
import joblib

from src.data_preprocessing import cap_exam_score
from src.feature_engineering import build_preprocessor
from src.model_training import train_and_log_model

SEED = 42

def run_training_pipeline(data: pd.DataFrame) -> dict:
    """Exécute le pipeline complet et retourne les métriques."""
    df = cap_exam_score(data.copy(), max_score=100)
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    model = LinearRegression()
    results = train_and_log_model(
        model, "LinearRegression_Final",
        X_train_prep, X_test_prep, y_train, y_test
    )

    # Sauvegarde du pipeline complet
    final_pipe = SkPipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    final_pipe.fit(X_train, y_train)
    joblib.dump(final_pipe, 'models/final_pipeline.pkl')

    return results