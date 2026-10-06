import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "clean_text"
SCORING = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def get_classifier(name, random_state=42):
    """Retorna el classificador corresponent a `name`."""
    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=random_state, n_jobs=-1),
    }
    if name not in classifiers:
        raise ValueError(f"Model '{name}' desconegut. Opcions: {sorted(classifiers)}")
    return classifiers[name]


def get_count_columns(df):
    """Columnes de comptatges generades a la fase 1 (n_words, n_emojis...)."""
    return [col for col in df.columns if col.startswith("n_")]


def build_pipeline(model_name, count_columns, max_features=3000, random_state=42):
    """Construeix el Pipeline complet: features + classificador."""
    features = ColumnTransformer(
        [
            # token_pattern=r"\S+" perquè el text ja està net i així no es perden els emojis
            ("tfidf", TfidfVectorizer(max_features=max_features, token_pattern=r"\S+"), TEXT_COL),
            ("counts", MinMaxScaler(), count_columns),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", get_classifier(model_name, random_state)),
        ]
    )


def load_xy(path, label_col):
    """Llegeix un CSV de train/test i el separa en X i y."""
    df = pd.read_csv(path)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    count_columns = get_count_columns(df)
    return df[[TEXT_COL] + count_columns], df[label_col], count_columns


def train_and_evaluate(model_name, X, y, count_columns, max_features=3000, cv_folds=5, random_state=42):
    """Validació creuada sobre train i entrenament final amb tot el train."""
    pipeline = build_pipeline(model_name, count_columns, max_features, random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
    metrics = {f"cv_{name}": round(float(scores[f"test_{name}"].mean()), 4) for name in SCORING}
    metrics["cv_f1_macro_std"] = round(float(scores["test_f1_macro"].std()), 4)

    pipeline.fit(X, y)
    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrena un model.")
    parser.add_argument("--model", required=True,
                        help="logreg | naive_bayes | linear_svm | random_forest")
    parser.add_argument("--train", required=True, help="CSV de train")
    parser.add_argument("--model-out", required=True, help="On guardar el model (.joblib)")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y, count_columns = load_xy(args.train, args.label_col)
    logger.info("Entrenant '%s' amb %d files", args.model, len(X))

    pipeline, metrics = train_and_evaluate(
        args.model, X, y, count_columns,
        max_features=args.max_features,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    logger.info("Mètriques (validació creuada): %s", metrics)
    absolute_parent = Path(args.model_out).resolve().parent
    absolute_parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    logger.info("Guardat %s", args.model_out)
    
    

if __name__ == "__main__":
    main()