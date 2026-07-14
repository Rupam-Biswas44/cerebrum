"""
Machine Learning Service

Wraps Scikit-Learn, XGBoost, LightGBM, Optuna, and MLflow for automated model training,
experiment tracking, hyperparameter optimization, and export.
"""

import uuid
from typing import Any

import mlflow
import optuna
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

logger = structlog.get_logger(__name__)

# Fallback imports if libraries are missing in environment
try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


class MLService:
    """Service for running automated ML tasks with tracking."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def train_automl(
        self,
        project_id: uuid.UUID,
        experiment_name: str,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        n_trials: int = 5,
    ) -> dict[str, Any]:
        """
        Run a simple AutoML pipeline using Optuna to select the best model and hyperparameters.
        Logs all trials and the best model to MLflow.
        """
        logger.info("ml.train.start", project=str(project_id), target=target_column)

        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in dataframe.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # For simplicity, convert all categorical to string or dummy
        # In a real system, we'd have a robust preprocessing pipeline (ColumnTransformer)
        X = pd.get_dummies(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.set_experiment(f"{project_id}_{experiment_name}")

        def objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(nested=True):
                # Model selection
                models = ["RandomForest"]
                if xgb is not None:
                    models.append("XGBoost")
                if lgb is not None:
                    models.append("LightGBM")

                model_name = trial.suggest_categorical("model", models)

                if task_type == "classification":
                    if model_name == "RandomForest":
                        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("rf_max_depth", 2, 20)
                        model = RandomForestClassifier(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    elif model_name == "XGBoost" and xgb is not None:
                        n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("xgb_max_depth", 2, 10)
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    elif model_name == "LightGBM" and lgb is not None:
                        n_estimators = trial.suggest_int("lgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("lgb_max_depth", 2, 10)
                        model = lgb.LGBMClassifier(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")

                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="weighted")
                    mlflow.log_metric("val_f1", score)

                    # Log parameters
                    mlflow.log_params(trial.params)

                    return score

                elif task_type == "regression":
                    if model_name == "RandomForest":
                        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("rf_max_depth", 2, 20)
                        model = RandomForestRegressor(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    elif model_name == "XGBoost" and xgb is not None:
                        n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("xgb_max_depth", 2, 10)
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    elif model_name == "LightGBM" and lgb is not None:
                        n_estimators = trial.suggest_int("lgb_n_estimators", 10, 100)
                        max_depth = trial.suggest_int("lgb_max_depth", 2, 10)
                        model = lgb.LGBMRegressor(
                            n_estimators=n_estimators, max_depth=max_depth, random_state=42
                        )
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")

                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = mean_squared_error(y_val, preds)
                    mlflow.log_metric("val_mse", score)

                    # Log parameters
                    mlflow.log_params(trial.params)

                    return score

                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

        direction = "maximize" if task_type == "classification" else "minimize"
        study = optuna.create_study(direction=direction)

        with mlflow.start_run(run_name="AutoML_Optimization"):
            study.optimize(objective, n_trials=n_trials)

            best_value = study.best_value
            best_params = study.best_params

            mlflow.log_metric("best_score", best_value)
            mlflow.log_params({"best_" + k: v for k, v in best_params.items()})

            logger.info("ml.train.complete", best_score=best_value, best_params=best_params)

            return {
                "best_score": best_value,
                "best_params": best_params,
                "metric_name": "f1_score" if task_type == "classification" else "mse",
                "trials_run": n_trials,
            }
