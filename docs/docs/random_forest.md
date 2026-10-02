---
language:
  - en
license: other  # TODO: set the project's license
library_name: sklearn
pipeline_tag: text-classification
tags:
  - spam-detection
  - text-classification
  - tf-idf
  - scikit-learn
metrics:
  - accuracy
  - precision
  - recall
  - f1
---

# Random Forest for spam detection (`random_forest`)

A scikit-learn Random Forest classifier that labels a text message as spam or not spam, using TF-IDF features of the cleaned text plus simple counts of the original message. It is one of four models compared in the MLOps-Datatouille project.

## Table of Contents

- [Model Details](#model-details)
- [Uses](#uses)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation](#citation)
- [Glossary](#glossary)
- [More Information](#more-information)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)

## Model Details

### Model Description

Ensemble of decision trees, each trained on a bootstrap sample of the data and a random subset of features; the final prediction is the majority vote. It is the only non-linear model in the comparison.

The artifact is a single scikit-learn `Pipeline`, so feature extraction and classifier are stored and versioned together.

- **Developed by:** Berta Torrents, Diego Velilla, Alex Bueno, Joel Delgado, Carles Aguilera.
- **Funded by:** DataTouille. MLOPS Team.
- **Shared by:** Berta Torrents, Diego Velilla, Alex Bueno, Joel Delgado, Carles Aguilera.
- **Model type:** Random Forest (`sklearn.RandomForestClassifier`) on TF-IDF and count features; supervised binary text classifier.
- **Language(s) (NLP):** English
- **License:** TODO
- **Finetuned from model:** Not applicable (trained from scratch; no pretrained model is used).

### Model Sources

- **Repository:** DataTouille. MLOPS Team.
- **Paper:** Not applicable.
- **Demo:** Not applicable.

## Uses

### Direct Use

Classify a single English text message as spam or not spam. The input must go through the project's preprocessing (`src/data_preparation/preprocess.py`) to produce the `clean_text` column and the count columns that the pipeline expects; see [How to Get Started with the Model](#how-to-get-started-with-the-model).

Intended users are the project team and the reviewers of the project.

### Downstream Use

The model can be plugged into a larger application (e.g. an API that filters incoming messages) or used as a baseline to compare against more complex models. It is not designed to be fine-tuned; to adapt it to new data, retrain it with `dvc repro`.

### Out-of-Scope Use

- Production spam filtering without further validation on data from the target platform.
- Messages in languages other than English: stop-word removal and stemming are English-specific.
- Automated decisions with consequences for users (blocking accounts, deleting messages) without human review.
- Detecting other kinds of unwanted content (hate speech, fraud, misinformation): the model was only trained to separate spam from non-spam.

## Bias, Risks, and Limitations

- Slower to train and much larger on disk than the linear models.
- Tends to be less effective than linear models on very sparse, high-dimensional text features.
- Less interpretable than a single linear model.
- Trained on a single dataset: performance on messages from other sources, periods or platforms is unknown, and spam changes over time (concept drift).
- Bag-of-words features ignore word order and context, so reworded or obfuscated spam (e.g. `fr3e`, `w1n`) can evade the model.
- The model may be less accurate for writing styles under-represented in the training data (dialects, non-native English, heavy slang or emoji use), so legitimate messages from some groups could be flagged more often.
- **False positives** hide legitimate messages from the user; **false negatives** expose the user to spam or phishing.
- Messages can contain personal data; the dataset and the predictions must be handled accordingly.

### Recommendations

- Evaluate the model on data from the target platform before any real use, and monitor it over time for drift.
- Decide the acceptable balance between false positives and false negatives for the application, and review flagged messages with a human when the consequences matter.
- Retrain periodically with recent data.
- Do not use the model outside English messages.

## Training Details

### Training Data

`daisy_dataset_spam_detection.csv` (TODO: where the dataset comes from (URL / provider) and its license), versioned with DVC. After duplicate removal and a stratified train/test split, the training set contains TODO messages:

| Class | Messages | Share |
|---|---|---|
| spam | TODO | TODO |
| not spam | TODO | TODO |

### Training Procedure

The whole procedure is a DVC pipeline (`dvc.yaml`) with three stages: cleaning, splitting and training.

#### Preprocessing

Implemented in `src/data_preparation/`:

1. Exact duplicate removal on the original text.
2. Count features on the original text: `n_characters`, `n_words`, `n_sentences`, `n_uppercase`, `n_urls`, `n_emojis`, `n_punctuation`.
3. Text cleaning: HTML tags removed, URLs replaced by a `[URL]` token, money amounts replaced by a `[DINERO]` token, lowercasing, punctuation and digits removed, English stop words removed, Porter stemming. **Emojis are deliberately kept** as tokens.
4. Second duplicate removal on the cleaned text.
5. Stratified train/test split.

Inside the model pipeline, the cleaned text is vectorised with TF-IDF (whitespace tokenisation, vocabulary limited to 3000 terms; TODO terms learned) and the counts are min-max scaled to [0, 1].

#### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| `n_estimators` | `300` |
| `max_depth` | `None` |
| `max_features` | `sqrt` |
| `class_weight` | `balanced` |

Other settings (TF-IDF vocabulary size, number of folds, random seed) are in `params.yaml`.

#### Speeds, Sizes, Times

- **Model size on disk:** TODO
- **Training time:** TODO (seconds on a laptop CPU)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The results below come from **stratified k-fold cross-validation on the training set**. The TF-IDF vocabulary and the scaler are re-fitted inside each fold, so there is no leakage between folds.

The held-out test set produced by the split stage has **not** been used for these numbers: it is reserved for the final evaluation of the selected model.

#### Factors

Characteristics expected to influence the model's behaviour: message length, presence of URLs, emojis, uppercase text and punctuation, vocabulary not seen during training, language, and the class imbalance between spam and non-spam. Results have not been disaggregated by these factors.

#### Metrics

Accuracy, precision, recall and F1, **macro-averaged** so that both classes weigh the same regardless of their size. F1 (macro) is the main metric because the classes are imbalanced and both kinds of error matter.

### Results

Values from `metrics/random_forest.json`:

| Metric | Cross-validation mean |
|---|---|
| Accuracy | TODO |
| Precision (macro) | TODO |
| Recall (macro) | TODO |
| F1 (macro) | TODO |

Standard deviation of F1 (macro) across folds: TODO.

Comparison with the other models (F1 macro):

| Model | F1 (macro) |
|---|---|
| logreg | TODO |
| naive_bayes | TODO |
| linear_svm | TODO |
| **random_forest** | TODO |

#### Summary

TODO: one or two sentences on how this model compares with the others.

### Societal Impact Assessment

No formal assessment of societal harm has been carried out. The main foreseeable harms are described in [Bias, Risks, and Limitations](#bias-risks-and-limitations).

## Model Examination

Strengths of this model:

- Captures non-linear relationships and interactions (e.g. between counts and specific words).
- Provides feature importances and probability estimates.

No dedicated interpretability analysis has been performed yet.

## Environmental Impact

- **Hardware Type:** TODO (e.g. laptop CPU)
- **Hours used:** TODO (training takes seconds to a few minutes)
- **Cloud Provider:** Not applicable (trained locally).
- **Compute Region:** Not applicable.
- **Carbon Emitted:** Not measured.

## Technical Specifications

### Model Architecture and Objective

`Pipeline(features -> classifier)`, where `features` is a `ColumnTransformer` combining `TfidfVectorizer` on `clean_text` and `MinMaxScaler` on the count columns, and `classifier` is `RandomForestClassifier`.

Each tree greedily minimises Gini impurity on a bootstrap sample; the forest predicts by majority vote.

### Compute Infrastructure

#### Hardware

Any modern CPU; no GPU required.

#### Software

Python, scikit-learn TODO, pandas, NLTK, emoji, joblib, DVC. See `requirements.txt`.

## Citation

Not applicable: there is no associated publication.

## Glossary

- **TF-IDF:** Term Frequency - Inverse Document Frequency; scores how important a word is to a message relative to the whole collection.
- **Stemming:** reducing words to a root form (e.g. "running" -> "run").
- **Stop words:** very frequent words ("the", "and") that carry little meaning.
- **Cross-validation:** splitting the training data into k parts, training on k-1 and validating on the remaining one, k times.
- **Macro average:** the metric is computed for each class and then averaged, giving both classes the same weight.

## More Information

The repository contains the full pipeline (`dvc.yaml`, `params.yaml`), the unit tests of the preprocessing functions (`test/`) and the model cards of the other models (`model_cards/`).

## Model Card Authors

TODO: team members

## Model Card Contact

TODO: contact e-mail or repository URL

## How to Get Started with the Model

```python
import joblib
import pandas as pd
from src.data_preparation import preprocess as pp

model = joblib.load("models/random_forest.joblib")

df = pd.DataFrame({"text": ["WIN a FREE prize!!! 🎉 http://spam.com"]})
df = pp.add_count_features(df, "text")
df = pp.add_clean_text(df, "text")
print(model.predict(df))
```

To reproduce the model:

```bash
dvc pull
dvc repro train@random_forest
dvc metrics show
```
