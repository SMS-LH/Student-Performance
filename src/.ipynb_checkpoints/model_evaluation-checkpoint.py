"""Métriques de régression."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Retourne un DataFrame avec MAE, RMSE, MAPE, R² (train/test)."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    data = {
        'Modèle': model_name,
        'MAE_train': mean_absolute_error(y_train, y_train_pred),
        'MAE_test': mean_absolute_error(y_test, y_test_pred),
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'RMSE_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAPE_train': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        'MAPE_test': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
        'R2_train': r2_score(y_train, y_train_pred),
        'R2_test': r2_score(y_test, y_test_pred)
    }
    return pd.DataFrame([data]).round(4)