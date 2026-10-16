# Proposal: Track energy and CO2 emissions with CodeCarbon

| Field | Value |
|---|---|
| **Scope** | How the team measures, stores and reports the energy consumption and CO2 emissions of the pipeline stages (Milestone 3, energy efficiency awareness). |
| **Out of scope** | Continuous resource monitoring of the deployed API (Milestone 6, Prometheus and Grafana), and the final choice of model, which is decided in experimentation. |

Official documentation: <https://docs.codecarbon.io/latest/>

---

## 1. Summary

This proposal adds a single shared helper, `track_emissions(stage)` in `src/emissions.py`, that
measures the energy (kWh) and CO2 emissions (kg CO2eq) of any block of code with CodeCarbon.
Each measurement is appended to a local CSV and, when an MLflow run is active, logged to MLflow
as metrics next to the model's quality metrics. Training is the only stage that must be
measured. Feature extraction and batch inference are recommended. The strongest reason is that
Milestone 3 evaluates **reflection**, not only measurement: placing energy next to F1 in the
same MLflow run is what makes it possible to compare models and justify decisions with numbers.

## 2. Context and motivation

Milestone 3 (Model building: quality assurance) requires *reporting CO2 emissions* with
CodeCarbon. Its evaluation criterion is: *how do your energy tracking practices help you reflect
on the efficiency and sustainability of your choices?* Measuring is therefore only half of the
work. The other half is **comparing** runs and **writing down what was decided because of the
numbers**.

At the time of this proposal, `src/features.py`, `src/modeling/train.py` and
`src/modeling/predict.py` are still the Cookiecutter placeholders. There is no real workload to
measure yet. The setup PR therefore only adds the shared infrastructure, and each stage owner
adds the tracker when writing the real code (section 6).

## 3. Goals and non-goals

**Goals:**

- Every training run logged to MLflow includes `train_duration_s`, `train_energy_consumed_kwh`
  and `train_emissions_kg_co2eq`.
- Every team member measures with the same settings (country, tracking mode, sampling), so
  results from the same machine are comparable.
- The tracking code is written once and reused by every stage, with one line per stage.
- The report contains a results table (energy, emissions and quality per model and stage), a
  decision justified with those numbers, and the limitations of the measurement.
- The setup works locally, in CI, in Docker and on the FIB VM without network access.

**Non-goals:**

- **Tracking each API request.** Starting and stopping a tracker per HTTP request adds latency
  and measures almost nothing. The API cost is estimated from batch inference or a load test
  instead (section 5.8).
- **Continuous monitoring of the deployed service.** Covered in Milestone 6 with Prometheus and
  Grafana.
- **Committing raw measurements to Git.** Raw CSVs stay local (section 5.6).
- **Real-time carbon intensity.** The country average is used (section 5.3).

## 4. Assumptions and constraints

| Assumption | Status | If it fails |
|---|---|---|
| CodeCarbon 3.x keeps the `OfflineEmissionsTracker` API used by the helper. | verified (3.3.1) | `requirements.txt` pins `codecarbon>=3.0`. Pin an exact version if a later release breaks it. |
| The `experiment_name` argument is **not** written to the CSV in 3.3.1, so the stage is stored in `project_name` as `mlops-datatouille-<stage>`. | verified (3.3.1) | If a later version writes it, nothing breaks, the stage simply also appears there. |
| Hardware power sensors are available (Intel RAPL on Linux, NVIDIA NVML for GPUs). | unknown, depends on each laptop | CodeCarbon falls back to an estimate based on the CPU's TDP. This happens on most macOS and Windows laptops and inside some VMs. The startup log says which method was used, and the report states it as a limitation. |
| MLflow is installed and a run is active when `track_emissions` is called. | assumed (MLflow is not yet in `requirements.txt`) | The helper skips MLflow logging silently. The CSV and the log line are still produced. |
| The team logs to a common MLflow server. | unknown (`infra/mlflow/` is a local Docker server per laptop) | Each member only sees their own runs. The final comparisons are run by one person on one machine, so the report is not affected. |
| Spain's average grid carbon intensity is representative of where the code runs. | assumed | If code runs elsewhere (e.g. Colab in another region), change `CODECARBON_COUNTRY_ISO_CODE` for those runs and state it in the report. |

- Paths and constants live in `src/config.py`, never hardcoded (CONTRIBUTING, AGENTS.md).
- Every new function in `src/` has a Pytest test, and CI runs `pytest tests/unit/` on PRs.
- The Cookiecutter layout is kept. No new packages or folders without team approval.
- Runs on different machines are **not** directly comparable. Any comparison in the report must
  come from runs on the same machine.

## 5. Design

### 5.1 Overview

| File | Status | Role |
|---|---|---|
| `requirements.txt` | edited (setup PR) | Adds `codecarbon>=3.0`. Everyone, including the CI runner, installs it with `pip install -r requirements.txt`. |
| `src/config.py` | edited (setup PR) | Adds `EMISSIONS_DIR` (`reports/emissions/`) and the `CODECARBON_*` constants. |
| `src/emissions.py` | new (setup PR) | Provides `track_emissions(stage)`, the only function the team calls. |
| `tests/unit/test_emissions.py` | new (setup PR) | Checks that a CSV row is written, including when the wrapped code crashes. |
| `reports/emissions/.gitkeep` | new (setup PR) | Keeps the empty folder in Git, since Git does not track empty folders. |
| `.gitignore` | edited (setup PR) | Ignores `reports/emissions/*.csv`. |
| `src/features.py`, `src/modeling/train.py`, `src/modeling/predict.py` | pending (stage owners) | Wrap the relevant block with `track_emissions(...)` (section 6). |

Flow of one measurement:

```
with track_emissions("train"):        # start tracker
    model.fit(...)                    # power sampled every 15 s
                                      # on exit (also on crash):
                                      #   1. stop tracker, compute kWh and kg CO2eq
                                      #   2. append row to reports/emissions/emissions.csv
                                      #   3. print log line
                                      #   4. if MLflow run active: log metrics + CSV artifact
```

### 5.2 Shared helper `src/emissions.py`

**What it does.** The module has four functions.

- `build_tracker(stage)` creates an `OfflineEmissionsTracker` with the settings from
  `src/config.py`. `stage` is a label (e.g. `"features"`, `"train"`, `"evaluate"`,
  `"predict"`) stored in the CSV `project_name` column as `mlops-datatouille-<stage>`.
- `emissions_summary(tracker)` extracts the three figures that matter from a stopped tracker:
  duration (s), energy (kWh) and emissions (kg CO2eq).
- `_log_to_mlflow(...)` logs those figures to the active MLflow run as `<stage>_duration_s`,
  `<stage>_energy_consumed_kwh` and `<stage>_emissions_kg_co2eq`, and attaches the CSV as an
  artifact under `emissions/`. If MLflow is not installed or no run is active, it does nothing.
- `track_emissions(stage)` is a context manager. It starts the tracker, runs the wrapped block,
  then stops the tracker, writes the CSV row, prints a log line such as
  `[train] 0.0021 kWh, 0.00036 kg CO2eq in 412.3 s`, and logs to MLflow. A `try/finally`
  guarantees that the measurement is saved even if the wrapped code crashes.

**Why.** One helper avoids each member configuring CodeCarbon differently, which would make
results incomparable. The `stage` label lets the same code be reused anywhere and keeps the
CSV rows and MLflow metrics of each stage apart. Using CodeCarbon only for training is valid,
since `stage` is just a label.

**Cost.** One extra dependency and one small module. The tracker runs a background sampler, so
it adds a negligible overhead to the measured block.

### 5.3 Offline mode with Spain's carbon intensity (`CODECARBON_COUNTRY_ISO_CODE = "ESP"`)

**What it does.** `OfflineEmissionsTracker` does not call the internet to geolocate the
machine. It uses Spain's average grid carbon intensity to convert kWh into kg CO2eq.

**Why.** It works inside CI, Docker and the FIB VM, where network access may be restricted.
Every member computes CO2 with the same factor, so results are comparable.

**Cost.** The CO2 figure uses the yearly country average, not the real-time electricity mix.
Real-time data would need an Electricity Maps API token.

### 5.4 Process tracking mode (`CODECARBON_TRACKING_MODE = "process"`)

**What it does.** Only the energy attributable to our Python process is measured, not the whole
machine.

**Why.** On a personal laptop, other programs (the browser, Spotify) would otherwise inflate
the figures.

**Cost.** The share of power attributed to the process is estimated from its CPU usage,
sampled at each reading. It is an estimate, not a direct measurement.

### 5.5 Power sampling every 15 s (`CODECARBON_MEASURE_POWER_SECS = 15`)

**What it does.** The CO2 total is computed once, at the end. The 15 s interval controls how
often **power** (watts) is read while the code runs. CO2 is obtained in two steps:

1. **Energy (kWh).** CodeCarbon cannot measure kWh directly. It reads the power of CPU, GPU and
   RAM every 15 s and accumulates power x time. When the tracker stops, it takes one last
   reading, so the time since the last 15 s mark is also counted.
2. **CO2 (kg).** At the end, total energy is multiplied by the carbon intensity (kg CO2 per
   kWh). In offline mode this factor is fixed, so no sampling is involved in this step.

**Why.** 15 s is frequent enough for runs of a few minutes and cheap enough not to distort them.
It is also CodeCarbon's default. It is kept as an explicit constant so the choice is visible and
can be stated in the report.

**Cost.** A shorter interval follows changes in load more closely (e.g. data loading vs. heavy
training) but adds overhead. A longer interval assumes the load stayed constant between
readings. The interval matters most in `process` mode. With hardware energy counters (Intel
RAPL) it matters less, because those counters accumulate energy between readings anyway.

### 5.6 Raw CSVs written locally but not committed

**What it does.** `EMISSIONS_DIR` in `src/config.py` tells CodeCarbon **where on the local
machine** to write `reports/emissions/emissions.csv`. The `.gitignore` rule
`reports/emissions/*.csv` tells Git **not to upload** that file. The file still exists locally
and can be opened. `reports/emissions/.gitkeep` keeps the folder in the repository.

**Why.**

- CodeCarbon appends a row on every run. If several members commit their own version of the
  same CSV, every PR conflicts on that file.
- Each row depends on the machine that ran it. A shared file would mix different machines in
  one table.
- The useful figures already go to MLflow, together with the quality metrics of each run.

**Cost.** Raw measurements of other members are only visible through MLflow, and only if the
team uses a common MLflow server (section 5.7). If the team wants
emissions visible in the repository for the teacher, a **summary table** built from the final
runs (e.g. `reports/emissions/summary.md`) can be committed. The `*.csv` rule does not block it.

No other ignore rule is needed. CodeCarbon does not create a `.codecarbon/` folder in the
project. It only writes a temporary lock file in the system temp folder
(`/tmp/.codecarbon.lock`).

### 5.7 MLflow as the record of each run

**What it does.** When `track_emissions` runs **inside** `mlflow.start_run(...)`, the energy and
emission metrics are stored in the same run as the parameters and quality metrics.

**Why.** MLflow's *Compare runs* view then gives a ready-made table and a scatter plot of F1 vs.
energy for the report, without extra code.

**Cost.** Order matters. If `track_emissions` is called outside an MLflow run, the metrics only
go to the local CSV. MLflow is only shared if everyone logs to the same server. The current
`infra/mlflow/` setup runs a local Docker server on each laptop, so each member only sees their
own runs. This is not a problem for the report, because valid comparisons must come from one
machine anyway (section 4). If the team deploys a common server (e.g. on the FIB VM), every
member sees every run. MLflow is imported only if installed, so the helper does not add MLflow as a
hard dependency.

### 5.8 Which stages are measured

| Stage | Label | Priority | Reason |
|---|---|---|---|
| Training (`src/modeling/train.py`) | `train` | **mandatory** | The minimum required by Milestone 3. Only the fitting step is wrapped, so data loading and evaluation do not inflate it. |
| Evaluation (`src/modeling/train.py`) | `evaluate` | recommended | Keeps the cost of evaluating separate from the cost of fitting. |
| Feature extraction (`src/features.py`) | `features` | recommended | If embeddings from a pretrained model are computed, this step can cost more than training the classifier. Without measuring it, that cost stays hidden. |
| Batch inference (`src/modeling/predict.py`) | `predict` | recommended | Training happens once, but the deployed model predicts on every email. Energy per 1,000 predictions is the recurring production cost, which dominates over time. It is also where a light model usually beats a transformer by far. |
| API (`src/api/`, Milestone 4) | none | **not tracked per request** | Per-request tracking adds latency and measures almost nothing. The batch inference figure is used instead, or a load test (e.g. 1,000 requests) is run under `codecarbon monitor` (section 6.5). |

## Verification

**Acceptance criteria.** This proposal succeeds if:

- `pytest tests/unit/test_emissions.py` passes in CI. The first test checks that one row is
  written with `project_name = mlops-datatouille-unit-test`, country `ESP`, and non-negative
  emissions and energy. The second checks that the row is still written when the wrapped block
  raises an error. Both write to a temporary folder (`tmp_path`), so they leave no CSV in the
  repository.
- Every final training run in MLflow has the three `train_*` emission metrics.
- The report contains the results table, the decision and the limitations listed below.

**Method.** No dedicated experiments are needed. The models compared during experimentation
are already trained inside MLflow runs, and `track_emissions("train")` adds their energy and
emissions next to their F1. The comparison for the report therefore comes from those same runs.
All runs used in the comparison are executed on the same machine.

The figures are different for every run, because they depend on the model, the data size and
the duration of the run. Repeating the same run on the same machine also gives slightly
different figures, because of background load, CPU temperature and sampling. If time allows,
each model is trained more than once and the mean is reported. Since every run uses the same
carbon factor, CO2 is energy multiplied by a constant, so ranking runs by CO2 or by energy gives
the same order.

**Reporting.** The person running the final experiments sends the report writer the following
information. If it is missing, the teacher cannot evaluate it.

- **Hardware.** CPU model, GPU model (if any) and RAM, taken from the CSV columns `cpu_model`,
  `gpu_model` and `ram_total_size`.
- **Where it ran.** Own laptop, FIB VM, Colab, etc., plus the OS.
- **Settings.** Offline mode, country `ESP`, `process` tracking mode, 15 s sampling, and the
  CodeCarbon version (`codecarbon_version` column).
- **Results table.** Per stage and per model, duration (s), energy (kWh), emissions (kg CO2eq)
  and the main quality metric.
- **Decision.** Which model or configuration was chosen and how the energy figures influenced it.
- **Limitations.**
  - Without hardware power sensors (Intel RAPL on Linux, NVIDIA NVML for GPUs), CodeCarbon falls
    back to an estimate based on the CPU's TDP. This happens on most macOS and Windows laptops
    and inside some VMs. The startup log says which method was used.
  - The CO2 figure uses Spain's average grid intensity, not the real-time mix.
  - Runs on different machines are not directly comparable.

The same figures go into the **model card** under an *Environmental impact* section (hardware,
hours used, compute region, carbon emitted), following the Hugging Face model card template.

## 6. Next Steps

Whoever writes the real code of each stage adds the tracker **in that same branch**, so nobody
touches the file twice and there are no merge conflicts.

### 6.1 Training (`src/modeling/train.py`), mandatory

Wrap **only the fitting step** with `track_emissions("train")`, inside the MLflow run. Evaluation
is tracked as a separate stage so each cost is visible.

```python
import mlflow

from src.emissions import track_emissions

with mlflow.start_run(run_name="logreg-tfidf"):
    mlflow.log_params(params)

    with track_emissions("train"):
        model.fit(X_train, y_train)

    with track_emissions("evaluate"):
        y_pred = model.predict(X_test)

    mlflow.log_metrics({"f1": f1_score(y_test, y_pred), ...})
```

`track_emissions` must be **inside** `mlflow.start_run(...)`, otherwise the metrics are only
written to the CSV and not to MLflow.

For a transformer fine-tuning with several epochs, the tracker also offers sub-tasks:

```python
with track_emissions("train") as tracker:
    for epoch in range(n_epochs):
        tracker.start_task(f"epoch_{epoch}")
        train_one_epoch(...)
        epoch_data = tracker.stop_task()
        mlflow.log_metric("epoch_emissions_kg_co2eq", epoch_data.emissions, step=epoch)
```

### 6.2 Feature extraction (`src/features.py`), recommended

```python
from src.emissions import track_emissions

with track_emissions("features"):
    features = build_features(df)
```

### 6.3 Batch inference (`src/modeling/predict.py`), recommended

Wrap the whole batch, then divide by the number of messages to report **energy per 1,000
predictions**.

```python
with track_emissions("predict") as tracker:
    preds = model.predict(X)
# per-message cost = tracker.final_emissions_data.energy_consumed / len(X)
```

### 6.4 MLflow dependency

MLflow is not yet in `requirements.txt`. Whoever adds experiment tracking adds it there, so the
MLflow logging of section 5.7 is active for everyone.

### 6.5 Zero-code alternative (notebooks, quick tests, API load tests)

Any command can be measured without editing it:

```bash
codecarbon monitor --no-api --offline --country-iso-code ESP -- python -m src.modeling.train
codecarbon detect   # shows which CPU/GPU/RAM CodeCarbon detects on your machine
```

The exact flags can change between versions. Check them with `codecarbon monitor --help`.