import os

os.environ["MODEL_NAME"] = "naive_bayes"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tests.perturb import append_text, perturb_dataframe
from tests.regression.test_robustness import (
    SPAM_PHRASE,
    model_inputs,
    spam_score,
)


model = joblib.load("models/naive_bayes.joblib")


test = pd.read_csv("data/dataset/test.csv")


perturbed = perturb_dataframe(test, text_col="text", seed=42)


f1_perturbed = f1_score(
    perturbed["label"],
    model.predict(model_inputs(perturbed)),
    average="macro",
)


unchanged = np.mean(
    model.predict(model_inputs(test))
    == model.predict(model_inputs(perturbed))
)


legitimate = test[test["label"] == "not_spam"]
extended = append_text(legitimate, SPAM_PHRASE, text_col="text")


non_decreasing = np.mean(
    spam_score(model, model_inputs(extended))
    >= spam_score(model, model_inputs(legitimate))
)


print(f"Macro F1 on perturbed messages: {f1_perturbed:.4f}")
print(f"Predictions unchanged after perturbation: {unchanged:.2%}")
print(
    f"Legitimate messages whose spam score does not decrease: "
    f"{non_decreasing:.2%}"
)