**Engineering Decision Notebook (EDN) — Entry**

**Milestone:** M2 & M3 (Model Building: Reproducibility and Quality Assurance)

**Activity / Topic (optional):** Hyperparameter tuning and configuration management

**Decision Participants:** All team members

**Decision:** Use Hydra only for hierarchical YAML config parsing and run the Optuna study loop natively inside `src/modeling/train.py`. One DVC `train` stage call (`model=${item}` plus DVC param overrides) executes the tuning loop, logs nested parent/child runs to MLflow, and saves the final optimal model to `models/<name>.joblib`.

**Rationale:** Alternative considered was Hydra's native `hydra-optuna-sweeper` plugin with `--multirun`. The sweeper launches trials as isolated independent processes, which clashes with DVC's DAG model: DVC expects the `train` stage to run a single deterministic command producing a final artifact. The selected approach preserves the DVC contract (single entrypoint, declared `conf/train.yaml` + `conf/model/*.yaml` deps, `models/*.joblib` outs), keeps MLflow parent/child trial tracking, and lets DVC-tracked params pass as Hydra overrides.

**AI Involvement:**

[x] No AI involvement

[ ] Information seeking

[ ] Alternative generation

[ ] Alternative assessment

[ ] Recommendation

[ ] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
