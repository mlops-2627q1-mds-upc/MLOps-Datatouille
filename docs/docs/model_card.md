---
language:
  - en
license: apache-2.0
library_name: sklearn
pipeline_tag: text-classification
tags:
  - spam-detection
  - naive-bayes
  - tf-idf
  - mlops
datasets:
  - diegovelilla/email-classification-dataset
metrics:
  - accuracy
  - precision
  - recall
  - f1
co2_eq_emissions:
  emissions: 0.000469
  source: CodeCarbon
  training_type: final fit with the selected hyperparameters
  geographical_location: Spain
  hardware_used: AMD Ryzen 7 4800H, 16 GB RAM
model-index:
  - name: message-classifier
    results:
      - task:
          type: text-classification
          name: Spam detection
        dataset:
          name: Email Classification Dataset (cleaned, stratified 80/20 test split)
          type: diegovelilla/email-classification-dataset
        metrics:
          - type: accuracy
            value: 0.9962
            name: Accuracy
          - type: precision
            value: 0.9962
            name: Macro precision
          - type: recall
            value: 0.9963
            name: Macro recall
          - type: f1
            value: 0.9962
            name: Macro F1
---

# Model Card for the Message Classifier

The Message Classifier is a supervised binary text classifier that labels an English message as `spam` or `not_spam`. It combines TF-IDF text features and seven message-level count features with a multinomial Naive Bayes classifier. It was selected among four candidate models because it obtained the highest cross-validation macro F1 score.

## Model Details

### Model Description

The model is a scikit-learn pipeline with two stages. The first stage builds the input features. A TF-IDF vectorizer with a vocabulary of at most 3,000 terms represents the normalized text. Seven count features computed on the original text are rescaled to the range [0, 1] with a min-max scaler. The second stage is a `MultinomialNB` classifier with an additive smoothing parameter of alpha = 0.1329, which was selected with Optuna. The complete pipeline is stored as a single serialized artifact, so the TF-IDF vocabulary, the scaler and the classifier are always loaded together.

The model is the spam detection component of a wider message-analysis system that also includes a summarization service based on an external large language model. This card documents only the spam classifier.

- **Developed by:** Team Datatouille (Carles Aguilera, Alex Bueno, Joel Delgado, Berta Torrents and Diego Velilla)
- **Funded by:** Not applicable. The model was developed as an academic project of the MLOps course of the Master in Data Science at UPC.
- **Shared by:** Team Datatouille, GitHub organization mlops-2627q1-mds-upc
- **Model type:** Supervised binary text classifier (multinomial Naive Bayes on TF-IDF and count features)
- **Language(s) (NLP):** English
- **License:** Apache 2.0, as stated in the `LICENSE` file of the repository
- **Finetuned from model:** Not applicable. The model is a classical machine learning classifier trained from scratch on the project data.

### Model Sources

- **Repository:** [https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille](https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille)
- **Paper:** Not applicable
- **Demo:** Not available at this delivery

## Uses

### Direct Use

The model receives a message and returns one of two labels, `spam` or `not_spam`. Because the classifier is probabilistic, it also returns the estimated probability of each class through `predict_proba`. Raw messages must first go through the same anonymization, count feature extraction and text normalization that were applied at training time. The section How to Get Started with the Model shows these steps.

### Downstream Use

The classifier is designed to be integrated into an intelligent message-analysis application. In that application, its prediction is combined with a message summary produced by an external large language model. The system design also includes the collection of user corrections, the monitoring of the prediction distribution and of data drift, and the automatic evaluation and retraining of the model when its performance degrades. These components belong to later milestones of the project and are not yet in operation at this delivery.

### Out-of-Scope Use

The model has not been validated on messages written in languages other than English, on domains that differ substantially from the training corpus, or on adversarial manipulations beyond the perturbations described in the Evaluation section. It is not intended to act as an authoritative fraud detector or as a security control with guaranteed detection. It should not be the only basis for deleting messages automatically. The model does not produce summaries.

## Bias, Risks, and Limitations

The origin and the labeling process of the source corpus are not documented. An exploratory analysis of the data showed that the legitimate messages resemble posts from public forums, whereas the spam messages appear to be synthetic. This observation is qualitative and does not describe a verified dataset generation process. The model may therefore learn stylistic differences that are specific to this corpus instead of general indicators of spam. For this reason, the very high scores on the test split do not guarantee a similar performance on real incoming email. The corpus contains only English messages and may contain mislabeled examples. Spam tactics also change over time, so the performance of the model may decrease under distribution shift. Very short messages, unusual writing styles, obfuscated links and deliberate typos can affect the predictions.

A false positive marks a legitimate message as spam, which may cause the user to miss it. A false negative lets an unwanted or harmful message through. The raw text may contain personal information. The preprocessing replaces recognized email addresses and phone numbers with placeholder tokens, but it does not guarantee the removal of every type of personal data. The probabilities returned by Naive Bayes are often poorly calibrated, so they should not be interpreted directly as real-world risk estimates.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Before the model filters messages automatically, it should be evaluated on a representative sample of real messages. The error rates of both classes should be monitored, false positives should be reviewed, and the performance should be reassessed periodically as message patterns change. The robustness of the model should be validated on other languages and writing styles before the supported scope is extended. Logs that contain user messages should be anonymized and access to them should be restricted. High-impact actions, such as deleting a message, should keep a human review step or a recovery option. The calibration of the probabilities should be assessed before they are presented to users as a confidence measure.

## How to Get Started with the Model

The trained pipeline is versioned with DVC and stored in the DagsHub remote of the project. Access to that remote requires a DagsHub account with access to the repository, configured as the credentials of the DVC remote `origin`. The following commands install the dependencies and download the data and the trained models.

```bash
git clone https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille.git
cd MLOps-Datatouille
pip install -r requirements.txt
dvc pull
```

The following code prepares raw messages with the preprocessing functions of the project and classifies them. It must be run from the root of the repository.

```python
import joblib
import pandas as pd

from src.data import preprocess as pp
from src.modeling.train import TEXT_COL, get_count_columns

model = joblib.load("models/naive_bayes.joblib")

messages = pd.DataFrame({"text": [
"WIN a FREE prize!! Visit http://example.com now",
"Hi team, the meeting has been moved to 3pm.",
]})

# Same steps as the training pipeline: anonymization, count features, text normalization.
messages["text"] = messages["text"].apply(pp.anonymize_text)
messages = pp.add_count_features(messages, "text")
messages = pp.add_clean_text(messages, "text", output_column=TEXT_COL)

features = messages[[TEXT_COL] + get_count_columns(messages)]
print(model.predict(features)) # predicted labels
print(model.predict_proba(features)) # class probabilities, in the order of model.classes_
```

## Training Details

### Training Data

The model was trained on the Spam Detection Dataset published by Deysi on Hugging Face (`Deysi/spam-detection-dataset`). The team republished it with documentation as the Email Classification Dataset (`diegovelilla/email-classification-dataset`), which is described in the [dataset card](https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille/blob/main/docs/docs/dataset_card.md). The original corpus contains 10,900 English messages, of which 5,500 are spam and 5,400 are not spam. After cleaning and deduplication, 10,629 messages remained. These messages were divided with a stratified 80/20 split and a fixed seed (42) into 8,503 training messages and 2,126 test messages. The original split of the source dataset was not reused, because duplicates crossed its partitions and would have leaked information from the training set into the test set.

### Training Procedure

The training procedure is implemented as a DVC pipeline with the stages `download`, `clean`, `validate_processed`, `split`, `train`, `evaluate` and `select`. The `train` and `evaluate` stages are executed once for each of the four candidate models. Every run is tracked in MLflow, and the experiment parameters are stored in `params.yaml`.

#### Preprocessing

The `clean` stage first anonymizes the messages. Email addresses and phone numbers are replaced with the tokens `xxemail` and `xxphone`. Exact duplicates are then removed. Seven count features are computed on the anonymized text before any normalization, so that information such as capitalization is preserved. These features are the number of characters, words, sentences, uppercase letters, URLs, emojis and punctuation marks.

The text is then normalized. HTML tags are removed and the text is lowercased. URLs and money amounts are replaced with the tokens `xxurl` and `xxmoney`. Punctuation, digits and English stop words are removed, and the remaining words are reduced to their stem with the Porter stemmer. The special tokens are not stemmed and emojis are kept as separate tokens. Messages that become empty are discarded, and duplicates of the normalized text are removed a second time.

The `validate_processed` stage checks the cleaned dataset with Great Expectations before it is split. The split is only produced if this validation passes.

#### Training Hyperparameters

- **Training regime:** CPU training in float64, the default precision of scikit-learn. No mixed precision was used.

The hyperparameters were tuned with Optuna inside the `train` stage. Hydra loads the search space of each model from `conf/model/`. For Naive Bayes, the only tuned hyperparameter was the smoothing parameter alpha, which was sampled on a logarithmic scale between 0.01 and 10. The study used the TPE sampler with seed 42 and ran 5 trials. Each trial evaluated the complete pipeline with stratified 5-fold cross-validation on the training split, and the trials were ranked by their mean macro F1 score. The best trial obtained alpha = 0.1329. A small alpha means that little probability mass is added to unseen term-class combinations, which is suitable when the vocabulary separates the classes well. The pipeline was finally refitted with this value on the complete training split.

The fixed parameters were a maximum TF-IDF vocabulary of 3,000 terms, 5 cross-validation folds and a random state of 42.

The same procedure was applied to logistic regression, linear SVM and random forest. The `select` stage compared the four models by their cross-validation macro F1 score, so the test split was not used for model selection. Naive Bayes obtained the highest score.

| Model | Cross-validation macro F1 |
|---|---:|
| Naive Bayes | **0.9978** |
| Logistic regression | 0.9972 |
| Linear SVM | 0.9972 |
| Random forest | 0.9955 |

The comparison scores in this table come from the separate energy benchmark described in the Environmental Impact section, which used default hyperparameters. Independently, the tuned Naive Bayes run achieved a cross-validation macro F1 of 0.9978 (standard deviation 0.0004). The benchmark and tuned training run are distinct evaluations.

#### Speeds, Sizes, Times

- **Final fit duration:** 0.7367 seconds on the benchmark laptop
- **Model artifact size:** 206,633 bytes (`models/naive_bayes.joblib`, including the TF-IDF vocabulary and the scaler)
- **Training samples:** 8,503
- **Test samples:** 2,126

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on the held-out test split of 2,126 messages, which keeps the class proportions of the cleaned dataset. This split was not used for hyperparameter tuning or for model selection. The robustness tests used perturbed copies of the same split.

#### Factors

The evaluation was disaggregated by two factors. The first factor is the class, through macro-averaged metrics that give the same weight to spam and to legitimate messages. The second factor is the robustness to changes in the writing of a message. The tests in `tests/regression/test_robustness.py` apply three perturbations that should not change the label. These perturbations are random changes of letter case (20% of the letters), typos made by swapping two adjacent letters (10% of the words with at least four letters), and obfuscated URLs (`http` written as `hxxp` and dots written as `[.]`). The count features and the normalized text are recomputed after each perturbation, so the model receives the modified messages exactly as it would in production. A directional test also appends a typical spam sentence to every legitimate message and checks that the spam probability does not decrease.

#### Metrics

The model was evaluated with accuracy, macro precision, macro recall and macro F1. Macro averages compute each metric for each class separately and then take the unweighted mean, so an error in either class has the same impact. The project requires the four metrics to reach at least 0.95 on the test split. The robustness tests require a macro F1 of at least 0.90 on the perturbed split, at least 90% of predictions unchanged after the perturbation, and a spam probability that does not decrease in at least 90% of the extended legitimate messages. These requirements are checked automatically by `tests/regression/` in the continuous integration pipeline.

### Results

| Metric | Value | Required | Evaluation data |
|---|---:|---:|---|
| Accuracy | 0.9962 | 0.95 | Test split |
| Macro precision | 0.9962 | 0.95 | Test split |
| Macro recall | 0.9963 | 0.95 | Test split |
| Macro F1 | 0.9962 | 0.95 | Test split |
| Macro F1 | 0.9978 | Not applicable | Stratified 5-fold cross-validation on the training split |
| Macro F1 on perturbed messages | 0.9934 | 0.90 | Perturbed test split |
| Predictions unchanged after perturbation | 99.72% | 90% | Test split and perturbed test split |
| Legitimate messages whose spam probability does not decrease | 100.00% | 90% | Legitimate messages of the test split |

#### Summary

The selected Naive Bayes classifier achieved a test accuracy of 0.9962, macro precision of 0.9962, macro recall of 0.9963, and macro F1 of 0.9962, exceeding the minimum requirement of 0.95 for all four metrics. The tuned cross-validation macro F1 was 0.9978 (standard deviation 0.0004 across five folds). The small difference between cross-validation and test scores is consistent with stable performance within this corpus, although it does not rule out overfitting or dataset-specific shortcuts. The robustness evaluation obtained a macro F1 of 0.9934 on perturbed messages, preserved 99.72% of predictions, and found non-decreasing spam probability for 100.00% of legitimate messages after a spam phrase was appended. All three robustness checks exceeded their specified thresholds. These results only describe the available corpus and the tested perturbations. They do not establish an equivalent performance on real email, on other languages or on future spam patterns.

## Model Examination

The `evaluate` stage extracts the 20 features with the largest absolute log-probability ratio between the spam and not-spam classes and stores them in `reports/features/naive_bayes.json`, versioned with DVC. A positive score indicates a stronger association with spam. A negative score indicates a stronger association with not spam. These scores describe class associations within the fitted model, not causal effects or calibrated changes in predicted probability.

| Feature | Log-probability ratio | More associated with |
|---|---:|---|
| `tfidf__dataset` | -6.385460 | Not spam |
| `tfidf__data` | -4.953725 | Not spam |
| `tfidf__rich` | 4.913405 | Spam |
| `tfidf__` (emoji) | 4.854064 | Spam |
| `tfidf__` (emoji) | 4.808488 | Spam |
| `tfidf__` (emoji) | 4.794611 | Spam |
| `tfidf__ampxb` | -4.724193 | Not spam |
| `counts__n_emojis` | 4.718357 | Spam |
| `tfidf__discount` | 4.589117 | Spam |
| `tfidf__project` | -4.584608 | Not spam |
| `tfidf__` (emoji) | 4.555211 | Spam |
| `tfidf__hi` | -4.476155 | Not spam |
| `tfidf__research` | -4.471015 | Not spam |
| `tfidf__file` | -4.469774 | Not spam |
| `tfidf__pill` | 4.458423 | Spam |
| `tfidf__exclus` | 4.424887 | Spam |
| `tfidf__` (emoji) | 4.423455 | Spam |
| `tfidf__amaz` | 4.313776 | Spam |
| `tfidf__hottest` | 4.307104 | Spam |
| `tfidf__appreci` | -4.289697 | Not spam |

Promotional terms such as `rich`, `discount`, `pill` and `hottest` are associated with spam, together with several emojis and the number of emojis in the message. Technical and discussion-oriented terms such as `dataset`, `data`, `project`, `research` and `file` are associated with not spam. The terms `exclus`, `amaz` and `appreci` are stems produced by the Porter stemmer. The strongest feature of the whole model is the word `dataset`, which has no direct relation with spam. This result indicates a possible dataset shortcut, since the classifier may distinguish the topics and styles of the two sources of the corpus in addition to general spam characteristics. The term `ampxb` is not a real word. It is the residue of double-escaped HTML entities such as `&amp;#xB;`, because `remove_html` removes HTML tags but does not decode HTML entities. After the removal of punctuation and digits, these entities become an artificial token that reflects the formatting of the source texts and not their content. These findings are consistent with the limitations described in the Bias, Risks, and Limitations section and justify an evaluation on real email before deployment. The same feature analysis is produced for the other candidate models and is linked to the corresponding MLflow runs.

## Environmental Impact

Energy consumption and emissions were measured with [CodeCarbon](https://codecarbon.io/) instead of the Machine Learning Impact calculator, because CodeCarbon measures the actual training process. CodeCarbon was run in offline mode with the average carbon intensity of the Spanish electricity grid, and in process mode, so only the energy of the training process was counted.

- **Hardware Type:** Laptop with an AMD Ryzen 7 4800H CPU (16 threads) and 16 GB of RAM, running Linux. No GPU was used.
- **Hours used:** 0.00020464 hours (0.7367 seconds) for the final fit
- **Cloud Provider:** None. The model was trained on a local machine.
- **Compute Region:** Spain
- **Carbon Emitted:** 0.000469 g CO2eq (4.69 × 10⁻⁷ kg CO2eq) for the final fit, with an energy consumption of 2.7 mWh (2.7 × 10⁻⁶ kWh)

According to the confirmed MLflow run values, these figures correspond to the `final` fit with alpha = 0.1329 inside the DVC pipeline, not the Optuna trials or the full training stage. In the pipeline, CodeCarbon reads the power every 15 seconds, so a fit that lasts less than one second relies on very few readings and the value is only indicative. The Optuna trials were measured separately and logged in MLflow.

For a reliable comparison between models, a separate benchmark (`src/modeling/measure_co2.py`) was run on the same laptop. Each model was trained with its default hyperparameters (alpha = 1.0 for Naive Bayes) through 5-fold cross-validation and a final fit. This procedure was repeated 5 times per model, the models were interleaved with a cooldown pause to limit thermal effects, and the power was read every second. The results are stored in `reports/emissions/co2_runs.jsonl`.

| Model | Energy (mWh, mean ± std) | Emissions (mg CO2eq, mean ± std) | Duration (s, mean) |
|---|---:|---:|---:|
| Naive Bayes | 7.18 ± 0.21 | 1.25 ± 0.04 | 1.27 |
| Linear SVM | 7.06 ± 0.46 | 1.23 ± 0.08 | 1.41 |
| Logistic regression | 8.65 ± 2.06 | 1.51 ± 0.36 | 1.56 |
| Random forest | 49.19 ± 17.83 | 8.56 ± 3.10 | 6.79 |

Naive Bayes and linear SVM consumed a similar amount of energy, and the difference between them is smaller than their standard deviations. Naive Bayes was the most stable model across repetitions. Random forest consumed about seven times more energy than the other models without improving the macro F1 score. The benchmark figures cannot be compared directly with the final fit figures, because the benchmark includes the cross-validation folds and uses different hyperparameters and a different sampling interval. The measurements cover only the electricity of the training process and do not constitute a full life-cycle assessment.

## Technical Specifications

### Model Architecture and Objective

The model is a scikit-learn `Pipeline`. Its first step is a `ColumnTransformer` that applies a `TfidfVectorizer` (at most 3,000 terms, tokens split on whitespace so that emojis and one-letter words are kept) to the normalized text and a `MinMaxScaler` to the seven count features. The min-max scaling keeps all features non-negative, which multinomial Naive Bayes requires. The second step is a `MultinomialNB` classifier with alpha = 0.1329.

Multinomial Naive Bayes estimates, for each class, the probability of each feature from the training data, and adds alpha to every feature total to avoid zero probabilities. To classify a message, it applies Bayes' rule under the assumption that the features are conditionally independent given the class, and it returns the class with the highest posterior probability.

### Compute Infrastructure

The model was trained and measured on a single laptop. The pipeline runs on a CPU and does not need a GPU.

#### Hardware

AMD Ryzen 7 4800H CPU with 8 cores and 16 threads, 16 GB of RAM, Linux.

#### Software

The project is implemented in Python 3.12. The main libraries are scikit-learn 1.7.2 for the model, NLTK for stemming, Optuna 5.0.0 for hyperparameter tuning, Hydra for configuration, MLflow for experiment tracking, DVC for data, model and pipeline versioning, Great Expectations for data validation, CodeCarbon 3.3.1 for energy measurement, pytest for testing and Ruff for linting and formatting. The exact versions of all dependencies are listed in `requirements.txt`.

## Citation

**BibTeX:**

```bibtex
@misc{datatouille2026spam,
  title = {Message Classifier Model Card},
  author = {Aguilera, Carles and Bueno, Alex and Delgado, Joel and Torrents, Berta and Velilla, Diego},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille},
  note = {MLOps course project, Master in Data Science, UPC}
}
```

**APA:**

Aguilera, C., Bueno, A., Delgado, J., Torrents, B., & Velilla, D. (2026). *Message Classifier Model Card* [Machine learning model documentation]. GitHub. [https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille](https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille)

## Glossary

- **Spam:** Unwanted advertising or fraudulent message, as labeled in the dataset.
- **Not spam:** Message labeled as legitimate.
- **TF-IDF:** Term Frequency-Inverse Document Frequency. It represents a text as a vector in which each term is weighted by its frequency in the message and by its rarity in the corpus.
- **Multinomial Naive Bayes:** Probabilistic classifier that models the features of each class with a multinomial distribution and assumes that the features are conditionally independent given the class.
- **Additive smoothing (alpha):** Constant added to the feature counts so that no probability is zero.
- **Macro precision, recall and F1:** Unweighted mean of the per-class precision, recall and F1 score.
- **Cross-validation:** Procedure that divides the training data into folds and trains and evaluates the model several times, each time with a different fold held out.
- **Data drift:** Change in the distribution of incoming data with respect to the training data.
- **CO2eq:** Carbon dioxide equivalent, a unit that expresses the impact of greenhouse gases.

## More Information

- **Dataset card:** `docs/docs/dataset_card.md`
- **Pipeline definition and parameters:** `dvc.yaml`, `params.yaml` and `conf/`
- **Training, evaluation and selection code:** `src/modeling/`
- **Regression and robustness tests:** `tests/regression/`
- **Energy measurement:** `src/emissions.py`, `src/modeling/measure_co2.py` and `reports/emissions/`
- **Design decisions:** `docs/edn/`

## Model Card Authors

Carles Aguilera, Alex Bueno, Joel Delgado, Berta Torrents and Diego Velilla

## Model Card Contact

[datatouillefib@gmail.com](mailto:datatouillefib@gmail.com)