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

`dvc repro` also trains, but it does not create a DVC experiment, so its MLflow runs cannot be
restored.

## Restore an experiment

Find the experiment name in MLflow (run name prefix, or the `dvc.exp_name` tag). Then:

```bash
dvc exp pull origin <exp_name>
dvc exp apply <exp_name>
```

Your workspace now has the code, params and outputs of that experiment.
