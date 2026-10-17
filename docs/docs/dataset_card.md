---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: train
    num_examples: 8175
  - name: test
    num_examples: 2725
license: apache-2.0
task_categories:
- text-classification
language:
- en
pretty_name: Email Classification Dataset
tags:
- spam-detection
- text-classification
- email
source_datasets:
- Deysi/spam-detection-dataset
language_creators:
- found
annotations_creators:
- found
size_categories:
- 10K<n<100K
---

# Dataset Card for Email Classification Dataset

The Email Classification Dataset contains **10,900 English texts** labelled as `spam` or `not_spam`. 
It is used as training and evaluation data for the spam detection component of the MLOps course project at FIB-UPC.

## Dataset Details

### Dataset Description

This dataset contains 10,900 English messages for text classification tasks. 
The dataset is split into train (8,175 rows, 75%) and test (2,725 rows, 25%) splits. It 
contains two columns: `text` (full message content) and `label` (binary label [spam, not_spam]).

- **Curated by:** FIB-UPC MDS students, see [Dataset Card Contact](#dataset-card-contact). Source data by [Deysi](https://huggingface.co/datasets/Deysi/spam-detection-dataset).
- **Language(s) (NLP):** English
- **License:** Apache-2.0

### Dataset Sources

- **Original Repository:** https://huggingface.co/datasets/Deysi/spam-detection-dataset

Due to the original dataset lacking a dataset card, the sources that originated said dataset are unknown.

## Uses

### Direct Use

- **Binary spam classification:** The `label` column supports training and evaluating models that distinguish spam from legitimate (ham) messages.
- **Raw Text for LLM Training:** The `text` column provides raw English messages content suitable for language model pre-training or fine-tuning.

### Out-of-Scope Use

- Models trained on this dataset should **not** be expected to generalize to real-world phishing tactics, as the spam text contains exaggerated, promotional language that may not reflect actual phishing attempts.
- Performance on **non-English** emails is not tested and likely to be worse.

## Dataset Structure

### Dataset Instances

Each instance contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Full message text |
| `label` | `string` | Binary label `spam` or `not_spam` |

Example:

```json
{
  "text": "Get rich quick! Make millions in just days with our new and revolutionary system! Don't miss out on this amazing opportunity!",
  "label": "spam"
}
```

### Dataset Splits

| Split | Rows | % |
|-------|------|---|
| `train` | 8,175 | 75% |
| `test` | 2,725 | 25% |
| **Total** | **10,900** | **100%** |

The split was inherited unchanged from the source dataset.

### Label Distribution

| Label | Train | Test | Total |
|-------|-------|------|-------|
| `spam` | 4,125 | 1,375 | 5,500 |
| `not_spam` | 4,050 | 1,350 | 5,400 |

Text length ranges from 1 to 41,544 characters.

## Dataset Creation

### Curation Rationale

The dataset was selected to train the spam detection component of the MLOps course project in the Master in Data Science at FIB-UPC. 
It provides binary spam labels in English with a predefined train and test split. The source dataset was published without documentation. 
It was therefore republished under the team's account with a dataset card that documents its content, limitations and intended use within the project.

### Source Data

#### Data Collection and Processing

The original dataset was sourced from [Deysi/spam-detection-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset) on Hugging Face, 
containing 10,900 English emails with binary spam/not_spam labels. 

#### Who are the source data producers?

The source dataset was published on Hugging Face by the user Deysi. Its dataset card provides no information about the origin of the data.

### Annotations

#### Annotation process

The annotation process for the binary labels is unknown.

#### Who are the annotators?

The annotators of the binary labels are not documented in the source dataset.

### Personal and Sensitive Information

Anonymization not documented in the original dataset. However, the `not_spam` instances seem to be public forum posts. Some of them contain URLs, organisation names and details that authors disclosed about themselves, such as their employer or field of study. 
The `spam` instances appear to be synthetic and are not expected to contain real personal data. No anonymisation or PII audit was performed on the texts.

## Bias, Risks, and Limitations

- **Synthetic spam style.** Spam texts contain exaggerated, promotional language that may not reflect real-world phishing tactics. Models trained on this dataset may not generalize to real-life spam.
- **Temporal drift.** Dataset likely reflects spam patterns from its original (and unknown) collection period. Current spam tactics may differ.
- **English-only.** No multilingual coverage. Performance on non-English email is not tested and likely to be worse.

### Recommendations

Users should be made aware of the risks, biases, and limitations of the dataset. Specifically, users should:

- Validate models trained on this dataset on real email data before using them for email filtering, as neither the domain nor the spam style may generalize.
- Be aware that temporal and linguistic biases may limit applicability to other contexts.

## Citation

```bibtex
@misc{email_classification_2026,
  author    = {Aguilera, Carles and Bueno, Alex and Delgado, Joel and Torrents, Berta and Velilla, Diego},
  title     = {Email Classification Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/diegovelilla/email-classification-dataset}
}
```

**APA:**

> Aguilera, C., Bueno, A., Delgado, J., Torrents, B., & Velilla, D. (2026). Email Classification Dataset. Hugging Face. https://huggingface.co/datasets/diegovelilla/email-classification-dataset

Users of this dataset should also acknowledge the source dataset, [Deysi/spam-detection-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset).

## Glossary

- **Spam:** Unsolicited message, usually with promotional or fraudulent intent.
- **Ham:** Legitimate message that is not spam. It corresponds to the `not_spam` label.

## More Information

The code and documentation of the project are available in the team's repository, https://github.com/mlops-2627q1-mds-upc/MLOps-Datatouille.git

## Dataset Card Contact

Questions about the dataset can be addressed to **datatouillefib@gmail.com**. The project was carried out by the following team members.

- [Carles Aguilera](https://github.com/carles-aguilera-pilo)
- [Alex Bueno](https://github.com/AlexBuenoL)
- [Joel Delgado](https://github.com/Tostaza)
- [Berta Torrents](https://github.com/bertatorrents)
- [Diego Velilla](https://github.com/diegovelilla)
