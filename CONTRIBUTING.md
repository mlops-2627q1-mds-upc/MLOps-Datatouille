# Contributing Guidelines — Spam Detection ML System
Here you can find the documentation to maintain high code quality, system reproducibility, and good collaboration. All team members must follow this guide for every contribution.

---

## 1. Branching Strategy & Architecture: Two-Tiered Staged Flow

To support our dual-component ML architecture (primary spam classifier and secondary text analysis model) alongside continuous feedback loops, our team extends basic GitHub Flow into a two-staged architecture:

* `main` (Production):
  - Always stable, deployable, and production-ready.
  - Direct pushes and branch deletions are strictly forbidden via GitHub Rulesets.
  - Only accepts Pull Requests originating from dev. Direct PRs from task branches are blocked at the CI level.
  - Merges require passing the Comprehensive Pre-Production Validation workflow and at least one peer approval.
    
* `dev` (Integration & Staging Server):
  - Simulates our internal development and integration server.
  - Direct pushes and branch deletions are strictly forbidden via GitHub Rulesets.
  - Serves as the base branch for all feature, bugfix, and chore development.
  - Merges require passing the Fast Feedback & Code Integration workflow and at least one peer review.

* Task branches (`feat/*`, `fix/*`, etc.):
  - Short-lived branches created directly from dev.
  - Deleted immediately after squashing and merging into dev.

---

## 2. Step-by-Step Contribution Cycle

### Step 1: Claim or Create an Issue & Branch
Before writing code:
1. Check the project coordination board (Github's Project section).
2. (Create) and assign the task to yourself and mark it as *In Progress*. Note the issue ID (e.g., `#12`).
3. Create the branch from the same issue.

Branch Naming Convention:
* feat/12-spam-preprocessing
* fix/15-mlflow-uri-leak
* docs/3-dataset-card
* refactor/8-clean-tokenizer

### Step 2: Local Development & Data Isolation
1. Fetch new branch locally:
```bash
git fetch origin
git checkout branch-name
```
2. Virtual Environment: Ensure your virtual environment is activated (source venv/bin/activate).
3. Data & Model Hygiene (DVC):
    * Never commit large files (.csv, .parquet, .pkl, .onnx, .pt) directly to Git.
    * Use DVC to track data dumps or model artifacts:
    ```bash
    dvc add data/raw/spam_dataset.csv
    git add data/raw/spam_dataset.csv.dvc data/raw/.gitignore
    dvc push
    ```

### Step 3: Crafting Standard Git Commits
We adhere strictly to the 7 Rules of Git Commit Messages:
1. Separate subject from body with a blank line.
2. Limit the subject line to 50 characters.
3. Capitalize the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood (e.g., "Add feature", "Fix bug", not "Added" or "Fixes").
6. Wrap the body at 72 characters.
7. Explain what and why rather than how.

Example:
```
Add text cleaning pipeline for spam classification dataset

Implement regex normalization, emoji stripping, and tokenization
in src/features/build_features.py to prepare raw text for baseline
training. Closes #12.
```

Push changes to the remote feature branch:
```bash
git push -u origin <branch-name>
```

### Step 4: Open a Pull Request targeting `dev`
1. Open the PR targeting ```base: dev``` from ```compare: <branch-name>```.
2. Link the issue with Closes #<id>, and list tested items).
3. Request a review from at least one teammate.

### Step 5: Automated CI Gate (`dev`) & Peer Review
* **Automated Status Check:** GitHub Actions triggers the Fast Feedback & Code Integration job:
  - Enforces Flake8 and Pylint linting.
  - Executes unit tests under tests/unit/ (FastAPI route schemas, helper logic).
  - Executes lightweight Great Expectations schema checks on sample data.
* **Peer Review:** Reviewers inspect the code diff and verify that logic and formatting standards are maintained.

### Step 6: Squash and Merge
Our repository is configured to strictly enforce **Squash and Merge**.
* Once approved and all CI checks pass, click Squash and merge.
* Ensure the final squashed commit message adheres to the 7 Git commit rules, summarizing the entire PR.
* Delete the feature branch after merging to keep the repository clean:
    ```bash
    git checkout main
    git pull origin main
    git branch -d <branch-name>
    git push origin --delete <branch-name>
    ```

---

## 3. Release & Pre-Production Promotion (`dev` $\rightarrow$ `main`)
When a sprint milestone is complete and integrated within dev, code is promoted to main for release and production serving.
1. Open a Pull Request configuring: `base: main` $\$leftarrow `compare: dev`
2. **Automated CI Gate (`main`)**: GitHub Actions triggers the Comprehensive Pre-Production Validation job:
   - Source Verification: Fails immediately if the PR originates from any branch other than `dev`.
   - Full Data Pipeline Validation: Executes complete Great Expectations suites against the full dataset to prevent the data from schema corruption.
   - Automated Regression Testing: Runs tests/regression/ using Pytest to ensure model accuracy, classification boundaries, and confidence scores exceed our baselines.
   - Multi-Component Integration: Validates full integration of the spam classifier, the secondary text component (summarizer or multilabel classifier), and endpoints.
3. **Approval**: Requires formal sign-off from team maintainers.
4. **Merge & Tag**: Merge into main and create a release tag for production tracking:
   ```bash
   git checkout main
   git pull origin main
   git tag -a v1.0.0 -m "Release Milestone v1.0.0"
   git push origin v1.0.0
   ```

## 4. Coding Good Practices

Follow these rules for all Python code in `src/`, `tests/`, and `notebooks/`.

* Keep lines short (max 127 chars to pass CI, aim for ~99).
* Use blank lines to separate logic:
    * 2 blank lines around top-level functions and classes.
    * 1 blank line around methods inside a class.
* Put spaces around operators (`x = 1`, not `x=1`) and after commas (`f(a, b)`, not `f(a,b)`).
* Use type hints for all function arguments and return values.
* Naming:
    * Use `snake_case` for functions and variables: `my_func`, `clean_text`, `spam_threshold`.
    * Do not use `camelCase`: `myFunc`, `cleanText` are not allowed.
    * Use `PascalCase` for classes: `SpamClassifier`.
    * Use `UPPER_CASE` for constants: `MAX_TOKENS = 512`.
    * Use clear names: `clean_text()` not `f1()`, `spam_threshold` not `x`.
* Keep functions short and with one task only. If a function exceeds ~50 lines, split it.
* Do not use magic numbers. Move constants to `src/config.py`.
* Do not hardcode paths. Use `pathlib.Path` and config values.
* Do not use `print()`. Use `logging` instead.
* Sort imports in 3 groups: stdlib, third-party, first-party (`src`). Remove unused imports.
  ```python
  import logging
  from pathlib import Path

  import pandas as pd
  from sklearn.model_selection import train_test_split

  from src.config import DATA_PATH
  ```
* Set random seeds (`random`, `numpy`, `sklearn`) for reproducibility.
* Add a Pytest test for each new function in `src/`.
* Do not commit dead code, commented code, or debug files.

### Function Documentation: Google Python Style

* Document all public functions, classes, and methods.
* Use Google Style docstrings.
* Include: short summary, `Args:`, `Returns:`, `Raises:` (if applicable).
* Keep summary in imperative mood and in one line.

Example:
```python
def clean_text(text: str, lowercase: bool = True) -> str:
    """Clean raw message text for spam classification.

    Args:
        text: Raw input message.
        lowercase: If True, convert text to lowercase.

    Returns:
        Cleaned text string.

    Raises:
        ValueError: If text is empty.
    """
    if not text:
        raise ValueError("text must not be empty.")
    ...
```
