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

# 1. Carregar el model Naive Bayes entrenat
model = joblib.load("models/naive_bayes.joblib")

# 2. Carregar el dataset de test
test = pd.read_csv("data/dataset/test.csv")

# 3. Generar versions modificades dels missatges
perturbed = perturb_dataframe(test, text_col="text", seed=42)

# 4. Calcular el Macro F1 sobre els missatges modificats
f1_perturbed = f1_score(
    perturbed["label"],
    model.predict(model_inputs(perturbed)),
    average="macro",
)

# 5. Calcular quantes prediccions es mantenen iguals
unchanged = np.mean(
    model.predict(model_inputs(test))
    == model.predict(model_inputs(perturbed))
)

# 6. Afegir una frase típica de spam als missatges legítims
legitimate = test[test["label"] == "not_spam"]
extended = append_text(legitimate, SPAM_PHRASE, text_col="text")

# 7. Comprovar si la probabilitat de spam augmenta o es manté
non_decreasing = np.mean(
    spam_score(model, model_inputs(extended))
    >= spam_score(model, model_inputs(legitimate))
)

# 8. Mostrar els tres resultats
print(f"Macro F1 on perturbed messages: {f1_perturbed:.4f}")
print(f"Predictions unchanged after perturbation: {unchanged:.2%}")
print(
    f"Legitimate messages whose spam score does not decrease: "
    f"{non_decreasing:.2%}"
)