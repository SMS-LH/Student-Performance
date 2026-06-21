"""Entraînement et tracking MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """Entraîne, évalue et logge le modèle sur MLflow. Retourne les métriques."""
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "MAE_train": mean_absolute_error(y_train, y_pred_train),
            "MAE_test": mean_absolute_error(y_test, y_pred_test),
            "R2_train": r2_score(y_train, y_pred_train),
            "R2_test": r2_score(y_test, y_pred_test)
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        metrics["run_id"] = mlflow.active_run().info.run_id
        return metrics