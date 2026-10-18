# Experiment tracking with DVC and MLflow

## Run and log an experiment

Train with `dvc exp run`. This logs one MLflow parent run per experiment, named
after the experiment, with one child run per model (named after the model) and, under it, one run
per Optuna trial (`trial-<n>`).

```bash
dvc exp run -n <exp_name>
```

Then push the experiment so others can restore it:

```bash
dvc exp push origin <exp_name>
```

### Runs outside `dvc exp run`

`dvc repro` (or running a stage module by hand) also trains, but it does not create a DVC
experiment, so its MLflow runs differ:

- There is no parent run. Each model gets a standalone run named after the model (e.g.
  `logreg`), with its Optuna trial runs nested under it.
- Train and test metrics end up in **different runs**: evaluation cannot tell which training run
  belongs to the same pipeline execution, so it starts a new standalone run with the test
  metrics instead of resuming the training one.
- The runs have no `dvc.exp_name` or `dvc.exp_ref` tags, so they cannot be restored.

Use `dvc repro` only for quick local checks, and `dvc exp run` for any run you want to keep.

### Carbon emissions

Training energy and CO2 are measured with CodeCarbon (`src/emissions.py`) and logged at every
level of the run tree:

| Run | Metrics | Artifact |
|---|---|---|
| Experiment (parent) | `train_*`: total of all models trained in the experiment | `emissions/emissions.csv`: one row per trial and final fit of every model |
| Model | `train_*`: tuning trials + final fit; `train_final_*`: final fit only | none |
| Trial | `train_trial_*`: that trial's cross-validation | none |

Each `*` is `duration_s`, `energy_consumed_kwh` and `emissions_kg_co2eq`. Models whose training
DVC skipped (cached) did not spend energy in the experiment, so they are not in its total.
Outside `dvc exp run` there is no parent run, so the model run gets its own `emissions.csv`.

The local `reports/emissions/emissions.csv` (gitignored) keeps every measurement ever made on
your machine and is not uploaded to MLflow.

## Restore an experiment

Find the experiment name in MLflow (parent run name, or the `dvc.exp_name` tag on the parent and model runs). Then:

```bash
dvc exp pull origin <exp_name>
dvc exp apply <exp_name>
```

Your workspace now has the code, params and outputs of that experiment.
