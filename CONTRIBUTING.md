# Contributing Guidelines — Spam Detection ML System
Here you can find the documentation to maintain high code quality, system reproducibility, and good collaboration. All team members must follow this guide for every contribution.

---

## 1. Core Workflow: GitHub Flow

We strictly adhere to **GitHub Flow**:
- The `main` branch is **always deployable** and production-ready.
- All development is conducted in short feature, bugfix, or task branches created directly from `main`.
- Direct commits to `main` are strictly prohibited (the branch is protected).
- All merges require a passing CI/CD test suite and a reviewed, approved Pull Request (PR).

---

## 2. Step-by-Step Contribution Cycle

### Step 1: Claim or Create an Issue
Before writing code:
1. Check the project coordination board (Github's Project section).
2. (Create) and assign the task to yourself and mark it as *In Progress*. Note the issue ID (e.g., `#12`).

### Step 2: Create a Feature Branch
Ensure your local `main` is completely up-to-date, then branch off:
```bash
git checkout main
git pull origin main
git checkout -b <type>/<issue-id>-<short-description>
```

Branch Naming Convention:
* feat/12-spam-preprocessing
* fix/15-mlflow-uri-leak
* docs/3-dataset-card
* refactor/8-clean-tokenizer

### Step 3: Local Development & Data Isolation
1. Virtual Environment: Ensure your virtual environment is activated (source venv/bin/activate).
2. Data & Model Hygiene (DVC):
    * Never commit large files (.csv, .parquet, .pkl, .onnx, .pt) directly to Git.
    * Use DVC to track data dumps or model artifacts:
    ```bash
    dvc add data/raw/spam_dataset.csv
    git add data/raw/spam_dataset.csv.dvc data/raw/.gitignore
    dvc push
    ```

### Step 4: Crafting Standard Git Commits
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

### Step 5: Open a Pull Request (PR)
1. Open the PR targeting ```base: main``` from ```compare: <branch-name>```.
2. Link the issue with Closes #<id>, and list tested items).
3. Request a review from at least one teammate.

### Step 6: Automated CI/CD Tests & Code Review
* **Continuous Integration:** GitHub Actions will automatically run the integration test suite (Pytest, Flake8) against your branch. These checks must pass before merging.   
* **Peer Review:** The reviewer tests the code locally or inspects the diff. If updates are needed, make additional commits to the same branch and push.

### Step 7: Squash and Merge
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