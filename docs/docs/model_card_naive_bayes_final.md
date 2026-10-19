# Datatouille Spam Classifier — Naive Bayes

The Datatouille Spam Classifier is a supervised binary text classification model developed by Team Datatouille as part of the MLOps-Datatouille project. It uses a Naive Bayes classifier within a text-processing pipeline to identify whether an English-language message is `spam` or `not_spam`. The selected model combines high reported classification performance with low measured training energy consumption.

## Table of Contents

• [Model Details](#model-details)
• [Uses](#uses)
• [Bias, Risks, and Limitations](#bias-risks-and-limitations)
• [Training Details](#training-details)
• [Evaluation](#evaluation)
• [Model Examination](#model-examination)
• [Environmental Impact](#environmental-impact)
• [Technical Specifications](#technical-specifications)
• [Citation](#citation)
• [Glossary](#glossary)
• [More Information](#more-information)
• [Model Card Authors](#model-card-authors)
• [Model Card Contact](#model-card-contact)
• [How to Get Started with the Model](#how-to-get-started-with-the-model)

# Model Details

## Model Description

This model performs binary spam detection on messages in English. Its inputs are prepared by a reproducible preprocessing pipeline, and its text representation uses TF-IDF features. The classifier belongs to the Naive Bayes family; the selected Naive Bayes classifier uses an additive smoothing parameter of **alpha = 0.1329**, selected through Optuna. The wider application is intended to combine spam detection with an external-LLM summarization service, but this model card covers **only the spam classifier**.

• **Developed by:** Team Datatouille: Carles Aguilera, Alex Bueno, Joel Delgado, Berta Torrents, and Diego Velilla.
• **Funded by:** No specific funding source documented; developed as an academic MLOps project at UPC.
• **Shared by:** Team Datatouille / mlops-2627q1-mds-upc.
• **Model type:** Supervised binary text classifier; Naive Bayes with TF-IDF-based text features and engineered count features.
• **Language(s):** English, as represented in the source dataset.
• **License:** Model artifact license not established by the provided report. The dataset card reports Apache-2.0 for the republished dataset; this does **not** automatically determine the model license.
• **Finetuned From Model:** Not applicable; a classical machine-learning classifier trained on project data, not a fine-tuned foundation model.
• **Version:** Initial delivery; model artifacts are versioned through DVC.

## Model Sources

• **Repository:** https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille
• **Dataset card:** https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille/blob/main/docs/docs/dataset_card.md

# Uses

## Direct Use

The classifier takes a message and predicts one of two classes: `spam` or `not_spam`. In the planned application, it also provides a confidence score for its prediction. Inputs must pass through the **same anonymization, normalization, feature engineering, and TF-IDF transformation** used at training time.

## Downstream Use

The classifier is designed as the spam detection component of an intelligent message-analysis system. Planned downstream capabilities include combining its predictions with an external-LLM message summarizer, collecting user corrections, monitoring prediction distributions and drift, and triggering evaluation and retraining when performance changes. These are **project goals**, not evidence that those services are already operating in production.

## Out-of-Scope Use

The model has not been validated for non-English text, adversarially manipulated messages outside the tested perturbations, or domains substantially different from its training corpus. It should not be treated as an authoritative fraud detector, a security control with guaranteed detection, or a sole basis for automatically deleting important messages. It is not a summarization model.

# Bias, Risks, and Limitations

The underlying corpus has an **undocumented origin and labeling process**. The project report observes that legitimate messages resemble public-forum posts and spam examples appear synthetic. A classifier may therefore learn stylistic differences specific to this corpus rather than generalizable spam indicators. Despite strong held-out scores, performance on real incoming email has not been established. The dataset contains only English messages and may contain mislabeled examples. Spam tactics evolve over time, so performance may deteriorate under distribution shift. Short messages, unusual writing styles, obfuscated URLs, typos, and other adversarial content can affect predictions.

False positives may cause legitimate messages to be flagged; false negatives may allow unwanted or harmful content through. Raw text may contain personal information. The cleaning pipeline replaces recognized email addresses and international-format phone numbers, but this does not guarantee removal of every type of personally identifiable information. Predicted probabilities, if exposed, should not automatically be interpreted as calibrated real-world risk estimates.

## Recommendations

Evaluate on representative real-world messages before enabling automatic filtering. Monitor both spam and non-spam error rates, review false positives, and periodically reassess performance as message patterns change. Validate robustness across languages and message styles before expanding the supported scope. Maintain privacy-preserving logging and restrict access to user-submitted content. For high-impact actions, keep a human review or recovery path. Assess probability calibration before presenting confidence as a measure of reliability.

# Training Details

## Training Data

The project uses the **Spam Detection Dataset** published by Deysi on Hugging Face, republished by the team with documentation as the **Email Classification Dataset**. The original corpus contains **10,900 English messages**: **5,500 spam** and **5,400 not_spam**. After cleaning and deduplication, **10,629 messages** remain. These are divided using a fixed-seed, stratified **80/20 split** into **8,503 training** and **2,126 test** messages. The source data's original split is not reused, reducing the risk that duplicates leak across training and testing.

## Training Procedure

### Preprocessing

The data pipeline first anonymizes recognized email addresses and international-format phone numbers using placeholder tokens. It removes duplicates and extracts count features such as character count, uppercase-letter count, URL count, and emoji count **before** text normalization. Text normalization removes HTML tags, lowercases text, substitutes URL and money patterns with tokens, removes punctuation, digits, and stop words, and applies stemming. Emojis remain as separate tokens. Duplicate normalized messages are removed again. The text is represented using TF-IDF, alongside the engineered features.

The processed data is validated with **Great Expectations** before splitting and model training. Data and model artifacts are versioned with **DVC**, and training experiments are tracked with **MLflow**. **Hydra** manages model-specific configurations, and **Optuna** tunes hyperparameters using fixed-seed stratified **5-fold cross-validation**. Trials are ranked by mean macro F1; the best trial is refitted on the full training split. The selection stage compares candidate models using cross-validation F1.

The project compares Naive Bayes, logistic regression, linear SVM, and random forest. **Naive Bayes is selected** based on the highest reported cross-validation macro F1 (**0.9978**). Optuna selected an additive smoothing parameter of **alpha = 0.1329** for the final Naive Bayes classifier. This parameter controls smoothing of estimated feature probabilities and helps avoid zero-probability issues.

### Speeds, Sizes, Times

• **Reported training duration:** **0.7367 seconds** for the reported training measurement.
• **Training samples:** 8,503; **held-out test samples:** 2,126.
• **Cross-validation:** Stratified 5-fold on the training split.
• **Hyperparameter tuning:** Optuna; selected smoothing parameter **alpha = 0.1329**.

# Evaluation

## Testing Data, Factors & Metrics

### Testing Data

The held-out test split contains **2,126** messages from the cleaned, stratified dataset. It is not used for hyperparameter tuning or selecting the best classifier. Tests also use controlled perturbations of the test messages to examine robustness.

### Factors

Relevant factors include message language, text length, spam-versus-not-spam class, message style and source, the presence of URLs or unusual tokens, and robustness to typos, capitalization changes, and URL obfuscation.

### Metrics

The evaluation protocol tracks **accuracy**, **macro precision**, **macro recall**, and **macro F1**. Macro averages weight the two classes equally. The project requires all four test metrics to reach at least **0.95**. Cross-validation macro F1 is used for model selection. Regression tests additionally assess performance under perturbations and prediction invariance.

## Results

| Metric | Reported value | Evaluation protocol |
|---|---:|---|
| Accuracy | **0.997** | Held-out test split |
| Macro precision | **0.997** | Held-out test split |
| Macro recall | **0.997** | Held-out test split |
| Macro F1 | **0.997** | Held-out test split |
| Cross-validation macro F1 | **0.9978** | Stratified 5-fold cross-validation on the training split |

### Summary

The selected Naive Bayes classifier achieved **99.7%** accuracy, macro precision, macro recall, and macro F1 on the held-out test split, as reported by the project team. Its cross-validation macro F1 of **0.9978** was the highest among the four compared models. The held-out test metrics satisfy the project's minimum threshold of **0.95**. However, these results reflect the available dataset and do not establish equivalent performance on real-world emails, unseen languages, or future spam patterns.

# Model Examination

The project specifies extraction and logging of the **20 most influential features** for each trained model, with the report versioned using DVC and linked to the MLflow experiment. This can help assess whether predictions rely on meaningful spam-related tokens rather than dataset artifacts.

# Environmental Impact

The project measures energy use and emissions with **CodeCarbon**, configured for offline estimation using the average carbon intensity of Spain's electricity grid. The reported energy and carbon figures below are **user-provided** and are treated as measurements for the selected run.

• **Hardware Type:** Project benchmark laptop with **AMD Ryzen 7 4800H CPU, 16 GB RAM, Linux**.
• **Hours used:** **0.00020464 hours** (converted from 0.7367 seconds).
• **Compute Region:** Spain for the offline grid-intensity assumption.
• **Energy consumed:** **2.7 × 10⁻⁶ kWh** (**2.7 mWh**).
• **Carbon Emitted:** **4.69 × 10⁻⁷ kg CO₂eq** (**0.469 mg CO₂eq**).
• **Measurement method:** CodeCarbon process-level energy accounting and emissions estimation.

**Measurement caveat:** The project report also gives an independent Naive Bayes benchmark of **7.18 ± 0.21 mWh** and **1.25 ± 0.04 mg CO₂eq**, averaged over five repetitions of a default-hyperparameter 5-fold-CV-plus-final-fit protocol. These figures should not be mixed with the single-run numbers above because the protocols may differ. CodeCarbon estimates can depend on hardware counters, CPU usage, sampling intervals, and grid-intensity assumptions; short-run measurements are especially sensitive to estimation and timing uncertainty. The project's emissions accounting is not a full lifecycle assessment.

# Technical Specifications

## Model Architecture and Objective

The system is a supervised binary classifier. A text-processing pipeline transforms messages into TF-IDF features and computes auxiliary message-level count features. A Naive Bayes estimator predicts `spam` or `not_spam`. Its learning objective is to estimate class-conditional feature distributions and use Bayes' rule to assign class scores under the estimator's conditional-independence assumptions. The selected additive smoothing hyperparameter is **alpha = 0.1329**.

## Compute Infrastructure

### Hardware

The report documents a Linux laptop with an AMD Ryzen 7 4800H processor and 16 GB RAM for comparative energy measurements.

### Software

The project is implemented in Python and uses scikit-learn-style model pipelines and TF-IDF processing, Optuna for tuning, Hydra for configuration, MLflow for experiment tracking, DVC for artifact and pipeline versioning, Great Expectations for data checks, pytest for testing, Ruff for linting and formatting, and CodeCarbon for energy measurements. Exact dependency versions are maintained in the repository and should be read from the lock/configuration files for a reproducible release.

# Citation

## BibTeX

```bibtex
@misc{datatouille2026spam,
  title        = {Datatouille Spam Classifier: Naive Bayes Model Card},
  author       = {{Team Datatouille}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille},
  note         = {MLOps project delivery}
}
```

## APA

Team Datatouille. (2026). *Datatouille spam classifier: Naive Bayes model card* [Machine learning model documentation]. GitHub. https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille

# Glossary

• **Spam:** Unwanted advertising or fraudulent message, as labeled in the project dataset.
• **Not spam:** Message labeled as legitimate.
• **TF-IDF:** Term Frequency–Inverse Document Frequency, a numerical text representation that emphasizes terms informative within a corpus.
• **Naive Bayes:** Family of probabilistic classifiers that uses class-conditional feature models with a conditional-independence assumption.
• **Macro precision / recall / F1:** Unweighted mean of the corresponding per-class metric.
• **Cross-validation:** Repeated training/evaluation over folds of the training data for tuning and model selection.
• **Data drift:** Change in the distribution of incoming data compared with the training/reference distribution.
• **CO₂eq:** Carbon dioxide equivalent, a unit for greenhouse-gas impact.

# More Information

• **Project repository:** https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille
• **Dataset card:** https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille/blob/main/docs/docs/dataset_card.md
• **Pipeline configuration:** `dvc.yaml`, `params.yaml`, and `conf/model/` in the repository.
• **Training and evaluation code:** `src/modeling/`.
• **Energy measurements:** `src/emissions.py` and `reports/emissions/`.
• **Initial project report:** *MLOps Project Report — Initial Delivery*, Team Datatouille, October 20, 2026.

# Model Card Authors

Team Datatouille: Carles Aguilera, Alex Bueno, Joel Delgado, Berta Torrents, and Diego Velilla.

# Model Card Contact

**Team contact:** datatouillefib@gmail.com

# How to Get Started with the Model

The repository contains the reproducible pipeline and the trained artifacts are managed with DVC. Use the documented environment setup and credentials, then retrieve the corresponding data/model artifacts and run the appropriate pipeline stage. The exact command for loading the **selected** serialized model and the correct preprocessing entry point must be checked against the repository's current README and exported artifact. An executable prediction snippet is intentionally omitted here to avoid suggesting an unverified API or bypassing preprocessing.
