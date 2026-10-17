"""
Training stage of the pipeline: hyperparameter tuning and training with Hydra, Optuna, and MLflow.

Each run tunes one model with a seeded Optuna study, refits the best hyperparameters on the
whole train set, and writes the model plus the CV metrics of the best trial. MLflow runs are
nested as DVC experiment > model > trial (see src.modeling.tracking).

Usage (orchestrated by DVC):
    python -m src.modeling.train model=logreg metrics_out=reports/metrics/train/logreg.json
"""

import json
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

from src.config import MLFLOW_EXPERIMENT_NAME
from src.data.preprocess import make_tfidf
from src.emissions import track_emissions
from src.modeling.tracking import log_dataset, start_model_run, start_trial_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "clean_text"
SCORING = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
TUNING_METRIC = "cv_f1_macro"
CV_METRICS_ATTR = "cv_metrics"


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


def build_pipeline(classifier: Any, count_columns: List[str], max_features: int) -> Pipeline:
    """
    Build the full pipeline: TF-IDF on text plus scaled count columns, then the classifier.

    Args:
        classifier: Unfitted scikit-learn classifier.
        count_columns: Metadata count features for the numerical scaler.
        max_features: Vocabulary size of the TF-IDF vectorizer.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    features = ColumnTransformer(
        [
            ("tfidf", make_tfidf(max_features), TEXT_COL),
            ("counts", MinMaxScaler(), count_columns),
        ]
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def run_study(
    cfg: DictConfig, X: pd.DataFrame, y: pd.Series, count_columns: List[str]
) -> optuna.Study:
    """
    Run a seeded Optuna study maximizing the CV F1 macro, logging each trial as an MLflow run.

    Must be called while the model's MLflow run is active, so trial runs nest under it.

    Args:
        cfg: The combined Hydra configuration dictionary.
        X: Feature dataset.
        y: Target labels.
        count_columns: List of metadata count features for the numerical scaler.

    Returns:
        The finished study. Each trial stores its CV metrics in the ``cv_metrics`` user attr.
    """

    def objective(trial: optuna.Trial) -> float:
        with start_trial_run(cfg.model.name, trial.number):
            classifier = get_classifier(
                cfg.model.name, trial, cfg.model.search_space, cfg.train.random_state
            )
            pipeline = build_pipeline(classifier, count_columns, cfg.train.max_features)
            cv = StratifiedKFold(
                n_splits=cfg.train.cv_folds, shuffle=True, random_state=cfg.train.random_state
            )

            with track_emissions(f"train_trial_{cfg.model.name}", metric_prefix="train_trial"):
                scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)

            metrics = {
                f"cv_{name}": round(float(scores[f"test_{name}"].mean()), 4) for name in SCORING
            }
            metrics["cv_f1_macro_std"] = round(float(scores["test_f1_macro"].std()), 4)
            trial.set_user_attr(CV_METRICS_ATTR, metrics)

            mlflow.log_params(trial.params)
            mlflow.log_metrics(metrics)
            return metrics[TUNING_METRIC]

    sampler = optuna.samplers.TPESampler(seed=cfg.train.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.train.n_trials)
    return study


def train_and_evaluate(
    cfg: DictConfig, X: pd.DataFrame, y: pd.Series, count_columns: List[str]
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Tune hyperparameters with Optuna and train the final pipeline with the best ones.

    Args:
        cfg: The combined Hydra configuration dictionary.
        X: Feature dataset.
        y: Target labels.
        count_columns: List of metadata count features for the numerical scaler.

    Returns:
        Tuple containing the optimal fitted Pipeline and the CV metrics of the best trial.
    """
    study = run_study(cfg, X, y, count_columns)

    best_trial = optuna.trial.FixedTrial(study.best_params)
    best_classifier = get_classifier(
        cfg.model.name, best_trial, cfg.model.search_space, cfg.train.random_state
    )
    final_pipeline = build_pipeline(best_classifier, count_columns, cfg.train.max_features)

    with track_emissions(f"train_final_{cfg.model.name}", metric_prefix="train_final"):
        final_pipeline.fit(X, y)

    return final_pipeline, study.best_trial.user_attrs[CV_METRICS_ATTR]


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """
    Write metrics as JSON so DVC can track them and later stages can read them.

    Args:
        metrics: Metric name to value.
        path: Output JSON file. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path="../../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Entry point for the training phase configured by Hydra.

    Args:
        cfg: The combined Hydra configuration dictionary.

    Returns:
        None.
    """
    X, y, count_columns = load_xy(cfg.data.train, cfg.data.label_col)
    logger.info("Starting Optuna search for '%s' with %d rows", cfg.model.name, len(X))

    params = {
        "model": cfg.model.name,
        "label_col": cfg.data.label_col,
        "max_features": cfg.train.max_features,
        "cv_folds": cfg.train.cv_folds,
        "random_state": cfg.train.random_state,
        "n_trials": cfg.train.n_trials,
    }
    with start_model_run(cfg.model.name, MLFLOW_EXPERIMENT_NAME) as run:
        log_dataset(X, y, Path(cfg.data.train), context="training")
        pipeline, metrics = train_and_evaluate(cfg, X, y, count_columns)
        classifier_params = pipeline.named_steps["classifier"].get_params()
        mlflow.log_params(params)
        mlflow.log_params({f"classifier.{key}": val for key, val in classifier_params.items()})
        mlflow.log_metrics(metrics)
        logger.info("Best cross-validation metrics: %s (MLflow run %s)", metrics, run.info.run_id)

    save_metrics(metrics, Path(cfg.metrics_out))
    logger.info("Saved %s", cfg.metrics_out)

    model_path = Path(cfg.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved final optimal model to %s", model_path)


if __name__ == "__main__":
    main()
