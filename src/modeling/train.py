"""
Phase 3 of the pipeline: training and hyperparameter tuning with Hydra, Optuna, and MLflow.

Usage (orchestrated by DVC):
    python -m src.modeling.train model=logreg
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import joblib
import mlflow
from omegaconf import DictConfig
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from src.data.preprocess import make_tfidf
from src.emissions import track_emissions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "clean_text"
SCORING = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def get_classifier(
    name: str, trial: optuna.Trial, model_cfg: DictConfig, random_state: int = 42
) -> Any:
    """
    Instantiate a classifier with hyperparameters sampled from Optuna.

    Args:
        name: Name of the classifier algorithm.
        trial: Optuna trial object for hyperparameter suggestion.
        model_cfg: Hydra configuration dict containing the search space boundaries.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        An instantiated scikit-learn classifier ready for training.

    Raises:
        ValueError: If the specified model name is not supported.
    """
    if name == "logreg":
        c_val = trial.suggest_float("C", model_cfg.C.low, model_cfg.C.high, log=model_cfg.C.log)
        return LogisticRegression(
            C=c_val, max_iter=1000, class_weight="balanced", random_state=random_state
        )
    elif name == "naive_bayes":
        alpha = trial.suggest_float(
            "alpha", model_cfg.alpha.low, model_cfg.alpha.high, log=model_cfg.alpha.log
        )
        return MultinomialNB(alpha=alpha)
    elif name == "linear_svm":
        c_val = trial.suggest_float("C", model_cfg.C.low, model_cfg.C.high, log=model_cfg.C.log)
        return LinearSVC(C=c_val, class_weight="balanced", random_state=random_state)
    elif name == "random_forest":
        n_estimators = trial.suggest_int(
            "n_estimators", model_cfg.n_estimators.low, model_cfg.n_estimators.high
        )
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown model '{name}'. Options: logreg, naive_bayes, linear_svm, random_forest"
        )


def get_count_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify the count columns generated during Phase 1 preprocessing.

    Args:
        df: The processed feature dataset.

    Returns:
        List of column names prefixed with 'n_'.
    """
    return [col for col in df.columns if col.startswith("n_")]


def load_xy(path: str, label_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load a CSV file and separate it into training features (X) and target array (y).

    Args:
        path: File path to the CSV dataset.
        label_col: Exact string name of the target class column.

    Returns:
        Tuple containing the feature DataFrame (X), target Series (y), and count column names.
    """
    df = pd.read_csv(path)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    count_columns = get_count_columns(df)
    return df[[TEXT_COL] + count_columns], df[label_col], count_columns


def train_and_evaluate(
    cfg: DictConfig, X: pd.DataFrame, y: pd.Series, count_columns: List[str]
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Execute Optuna hyperparameter study, log via MLflow, and train the final global pipeline.

    Args:
        cfg: The combined Hydra configuration dictionary.
        X: Feature dataset.
        y: Target labels.
        count_columns: List of metadata count features for the numerical scaler.

    Returns:
        Tuple containing the optimal fitted Pipeline and its final cross-validation metrics.
    """

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            classifier = get_classifier(
                cfg.model.name, trial, cfg.model.search_space, cfg.train.random_state
            )

            features = ColumnTransformer(
                [
                    ("tfidf", make_tfidf(cfg.train.max_features), TEXT_COL),
                    ("counts", MinMaxScaler(), count_columns),
                ]
            )
            pipeline = Pipeline([("features", features), ("classifier", classifier)])
            cv = StratifiedKFold(
                n_splits=cfg.train.cv_folds, shuffle=True, random_state=cfg.train.random_state
            )

            with track_emissions(f"train_trial_{cfg.model.name}"):
                scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)

            metrics = {
                f"cv_{name}": round(float(scores[f"test_{name}"].mean()), 4) for name in SCORING
            }
            metrics["cv_f1_macro_std"] = round(float(scores["test_f1_macro"].std()), 4)

            mlflow.log_params(trial.params)
            mlflow.log_metrics(metrics)
            return metrics["cv_f1_macro"]

    with mlflow.start_run(run_name=f"optuna_search_{cfg.model.name}"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg.train.n_trials)

        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_f1_macro", study.best_value)

        best_trial = optuna.trial.FixedTrial(study.best_params)
        best_classifier = get_classifier(
            cfg.model.name, best_trial, cfg.model.search_space, cfg.train.random_state
        )

        features = ColumnTransformer(
            [
                ("tfidf", make_tfidf(cfg.train.max_features), TEXT_COL),
                ("counts", MinMaxScaler(), count_columns),
            ]
        )
        final_pipeline = Pipeline([("features", features), ("classifier", best_classifier)])

        with track_emissions(f"train_final_{cfg.model.name}"):
            final_pipeline.fit(X, y)

        return final_pipeline, {"cv_f1_macro": study.best_value}


@hydra.main(version_base=None, config_path="../../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Entry point for the training phase configured by Hydra.

    Args:
        cfg: The combined Hydra configuration dictionary.

    Returns:
        None.
    """
    mlflow.set_experiment("spam-classifier")

    X, y, count_columns = load_xy(cfg.data.train, cfg.data.label_col)
    logger.info("Starting Optuna search for '%s' with %d rows", cfg.model.name, len(X))

    pipeline, metrics = train_and_evaluate(cfg, X, y, count_columns)
    logger.info("Best cross-validation metrics: %s", metrics)

    model_path = Path(cfg.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved final optimal model to %s", model_path)


if __name__ == "__main__":
    main()
