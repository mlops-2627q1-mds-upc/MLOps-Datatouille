**Engineering Decision Notebook (EDN) - Entry**

**Milestone:** M2: Model Building - Reproducibility

**Activity / Topic (optional):** Data splitting.

**Decision Participants:** All team members.

**Decision:** Discard the original 75/25 split of the dataset. The pipeline joins both original splits, cleans and deduplicates the messages, and then creates a new stratified 80/20 split with a fixed random seed.

**Rationale:** The original split was made before any deduplication, so identical messages could appear in both the training and the test split and inflate the test metrics. Splitting after deduplication prevents this leakage. Stratification keeps the class proportions equal in both splits, as required by REQ-NF-06. The proportion and the seed are defined in `params.yaml`, so the split is reproducible and can be changed without editing the code.

**AI Involvement:** Select all that apply.

[x] No AI involvement

[ ] Information seeking

[ ] Alternative generation

[ ] Alternative assessment

[ ] Recommendation

[ ] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Other Evidence (optional):** `src/download_dataset.py` (joins the original splits), `src/data/cleaning.py` (deduplication), `src/data/split_data.py` and `src/data/splitting.py` (stratified split), the `split` section of `params.yaml`, and the `split` stage in `dvc.yaml`.
