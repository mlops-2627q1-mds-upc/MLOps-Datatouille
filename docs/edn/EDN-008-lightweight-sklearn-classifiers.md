**Engineering Decision Notebook (EDN) - Entry**

**Milestone:** M2 (Model Building: Reproducibility)

**Activity / Topic (optional):** Choice of the spam classification models

**Decision Participants:** All team members

**Decision:** Classify messages with traditional machine learning models from scikit-learn, namely logistic regression, multinomial naive Bayes, a linear SVM, and a random forest, trained on TF-IDF and count features, instead of deep learning or transformer based models. 

**Rationale:**

The pipeline follows the approach of [AlHomidan et al](https://thesai.org/Downloads/Volume16No11/Paper_19-Improving_Spam_Detection_with_Feature_Engineering_and_Adaptive_Learning_Approaches.pdf), who classify spam with traditional models using TF-IDF vectors and count-based features. We saw that it was not necessary to deploy heavy AI models to perform this task. Instead, by using a good set of features and these lightweight models, it is possible to obtain good results. The benefits from deploying these models (e.g. logistic regression) are:
- Low resource requirements, which allows us as developers to easily train and deploy models using our workstations.
- Ease of use, as they are integrated in the scikit-learn python library.

**AI Involvement:**

[x] No AI involvement

[ ] Information seeking

[ ] Alternative generation

[ ] Alternative assessment

[ ] Recommendation

[ ] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
