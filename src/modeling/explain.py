"""Identification of the features that most influence the predictions of a trained model."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

TOP_N = 20


def classifier_scores(classifier: BaseEstimator) -> tuple[np.ndarray, str]:
    """Return one influence score per input feature of a fitted classifier.

    Args:
        classifier: Fitted naive Bayes, linear (logistic regression, linear SVM), or tree
            ensemble classifier.

    Returns:
        Tuple with the score of each feature and the type of score. Positive scores of the
        signed types push the prediction towards ``classifier.classes_[1]``.

    Raises:
        TypeError: If the classifier exposes none of the supported attributes.
    """
    if hasattr(classifier, "feature_log_prob_"):
        log_prob = classifier.feature_log_prob_
        return log_prob[1] - log_prob[0], "log_probability_ratio"
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_)[0], "coefficient"
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_), "importance"
    raise TypeError(f"Unsupported classifier: {type(classifier).__name__}")


def top_features(pipeline: Pipeline, n: int = TOP_N) -> dict:
    """List the ``n`` features with the largest absolute influence on a fitted pipeline.

    Args:
        pipeline: Fitted pipeline with a ``features`` step and a ``classifier`` step.
        n: Number of features to keep.

    Returns:
        Mapping with the type of score, the class favoured by positive scores, and the
        features sorted by decreasing absolute score, each with its name and score.
    """
    names = pipeline.named_steps["features"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    scores, score_type = classifier_scores(classifier)
    order = np.argsort(-np.abs(scores), kind="stable")[:n]
    return {
        "score_type": score_type,
        "positive_class": str(classifier.classes_[1]),
        "features": [
            {"feature": str(names[i]), "score": round(float(scores[i]), 6)} for i in order
        ],
    }
