# Proposal: Data preprocessing pipeline

| Field | Value |
|---|---|
| **Scope** | Data preprocessing pipeline steps |
| **Out of scope** | Implementation details, model and evaluation metrics |

---

## 1. Summary

This document aims to define the set of steps to be part of the main data processing pipeline. The data pipeline will take our source dataset as input and will output a feature matrix suitable for supervised classification.

As modeling and experimentation is carried out, additional features (and therefore, new steps) will be added to the pipeline. Some of the steps presented in this document are structured in two phases: If Phase I features provide reasonable results, implementing Phase II features might not be needed. Hence, a future version of this document must be rewritten with the definitive pipeline.

## 2. Context and motivation

This project contains a ML model for classification of spam/ham messages. To train this model, we have a dataset containing thousands of unstructured text and labels indicating whether they are spam or not [ADD LINK TO DATASET]().

There are many approaches to solve this problem, some of them using advanced models such as BERT. However, some of these tend to be resource-hungry and difficults their deployment in scenarios with low resources.

Other approaches to solve this problem are based on light-weight traditional ML classifiers (e.g. Logistic Regression, Support Vector Machine, Random Forest, etc.). These models require tabular data. So, we need to develop a data preprocessing pipeline that cleans original data, transforms unustructured text into a feature matrix and produces the artifacts needed for model training.

The approach and steps of the pipeline have been inspired from the paper *[Improving Spam Detection with Feature Engineering
and Adaptive Learning Approaches](https://thesai.org/Downloads/Volume16No11/Paper_19-Improving_Spam_Detection_with_Feature_Engineering_and_Adaptive_Learning_Approaches.pdf)*


## 3. Goals and non-goals

**Goals** — what a successful outcome looks like:

- Define and justify steps for further implementation

**Non-goals** — explicitly out of scope, and why:

- Define final features, model and evaluation metrics. These are considered in the experimentation phase, which is out of the scope of this document.

*DISCLAIMER: This pipeline will present a set of possible features. It is not definitive, since that depends on training results*

## 4. Assumptions and constraints

Main assumptions come from the input and output shape:
- Source data is binary-labeled unstructured text.
- Text does not contain metadata or other structural information
- Language is English
- Positive rate is ~50% (balanced data)
- The consuming models receive a dataframe of numerical features

## 5. Design

### 5.1 Overview

<!--
  The shape of the solution. A diagram, a schema, a table — whatever makes the
  structure obvious at a glance. Keep it simple enough to be correct.
-->

Phase | Step | Description |
|---|---|---|
| I | Remove duplicates | Remove exact text duplicates |
| I | Compute counts from original text | Compute counts such as number of *characters, words, sentences, uppercases, punctuation marks, urls, phone numbers, emojis*  |
| II | Lowercase text | Convert all words to lowercase |
| II | Remove unnecessary information | Remove punctuation marks and emojis |
| II | Remove stop words | Remove words without useful information such as *the, a, and, etc.* using a stop-word list |
| II | Apply stemming | Apply stemming to reduce words to their basic forms, reducing vocabulary size |
| II | Remove duplicates (II) | Removing duplicates after having applied all the previous transformations
| II | Compute TF-IDF (on N words) | Compute TF-IDF score on root forms of words (or a subset of them) |
| I | Train / Test split | Split dataset into train and test sets


### 5.2 Remove duplicates

Remove duplicated texts in the dataset. This is done as a preliminary-step at the beginning to avoid mainly two things:
- Data leakage: The same text belongs to the train and test datasets.
- Redundancy in train dataset: Training a model with duplicated data can affect the learning process

Phase I features only consist on different counts based on the original text, so no text encoding or processing is applie. So, this duplicate removal can only be done based on exact-coincidence text, which is why it is applied as a first step.

After Phase I first experiment iteration is conducted, Phase II can be implemented, which applies a more refined duplicate removal based on additional preprocessing steps in the pipeline.

### 5.3 Compute counts from original text

By inspecting spam messages at a glance some characteristics can be noticed: Spam messages apparently tend to have more punctuation marks (e.g "!!!"), imperative messages with uppercase letters, emojis, urls or links, etc. Derived from this empirical observation, a set of counts are computed as they might be useful signals for predicting the class.

In this step, we group computing the following counts per message and the reason behind that decision:
- **character / word / sentence:** Spam messages on average might have more characters / words / sentences.
- **uppercase characters:** Spam messages on average might have a larger number of uppercase characters to attract reader's attention (e.g. "CLAIM PRIZE")
- **links:** Spam messages on average might have more links or urls, since their purpose is to decive the reader.
- **emojis**: Some spam messages might use a high number of emojis in order to attract readers.
- **punctuation marks**: Same as uppercase characters (e.g "CLAIM PRIZE !!!")

### 5.4 Lowercase text

Once the previous counts have been computed, we can proceed to apply transformations that imply information loss. The first of them is converting all texts to lowercase. This steps allows for two things:
- Future duplicate removal in scenarios where two messages can have the same content and be apparenty different due to upper/lower case organization.
- Vocabulary reduction for more efficient TF-IDF computation in the future

### 5.5 Remove unnecessary information

Remove punctuation marks, emojis and urls. They do not provide any lexical meaning or information. The justification is the same as [7.4](#74-lowercase-text): reducing vocabulary size and allow more refined duplicate removal.

The removal of these information can be easily achieved using regex.

### 5.6 Remove stop words

Remove stop words such as "the", "and", "or", etc. which do not provide useful information and would add noise when computing TF-IDF due to being very frequent terms. The justification is the same as [7.4](#74-lowercase-text).

The removal of these information can be easily achieved using regex combined with a stop-word list. Notice this depends on the observed language used on messages (English)

### 5.7 Apply stemming

Stemming is applied to reduce multiple versions of a similar word to a canonical representation or root form. It is used to group similar terms by meaning and reduce vocabulary size. The goal of this is the same as [7.4](#74-lowercase-text).

### 5.8 Remove duplicates (II)

After all the "simplification" achieved on the previous steps, we are able to do a more refined duplicate removal as a previous step of computing TF-IDF scores on the resulting vocabulary (or a subset of it). 

### 5.9 Compute TF-IDF (on N words)

TF-IDF (Term Frequency - Inverted Document Frequency) is a measure of importance of a word to an specific document in a collection used in the area of Information Retrieval. Basically, the TF-IDF score of a word *w* in a document *d* is:
- ~0: *w* is not important in *d*. That could be because either *w* has low frequency in *d* (maybe it does not even appear), because *w* is a common word that appears in many documents or both.
- \>=1: *w* is important in *d*. That could be because either *w* has high frequency in *d*, because *w* is a very uncommon word across all documents or both.

In the end, the goal is to translate from unstructured text to a set of numerical features. This way, the models may be able to detect relationships between the words and the final label. Some options are adding as many features (columns) as words in our vocabulary (e.g. 3000) containing the count of each word in each document. Another option is the same, but instead of having the count, having a binary value indicating whether the word appears or not. They might provide good results too. However, the reason why we decide to use TF-IDF values is because these scores provide additional information, which is the notion of importance of a word in an specific document.

### 5.10 Train / Test split

Once all preprocessing and feature extraction has been applied, we split the dataset intro train and test sets to avoid data leakage:
- Train set will be used for training models and comparing them to obtain the best one
- Test set will be used to assess the final performance of the selected model.

## 6. Next Steps

1. Implement Phase I measures and train first iteration of models as exploratory work (can be done using notebooks)
2. Implement Phase II measures if results are not satisfying
3. Implement pipeline using scripts and dvc

