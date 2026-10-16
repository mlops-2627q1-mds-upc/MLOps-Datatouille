# Experiment tracking with DVC and MLflow

## Run and log an experiment

Train with `dvc exp run`. This logs one MLflow parent run per experiment, with one child run
per model and, under it, one run per Optuna trial, all prefixed with the experiment name.

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

## Restore an experiment

Find the experiment name in MLflow (run name prefix, or the `dvc.exp_name` tag). Then:

```bash
dvc exp pull origin <exp_name>
dvc exp apply <exp_name>
```

Your workspace now has the code, params and outputs of that experiment.
