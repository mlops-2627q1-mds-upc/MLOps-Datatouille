**Engineering Decision Notebook (EDN) - Entry**

**Milestone:** M2 & M3 (Model Building: Reproducibility and Quality Assurance)

**Activity / Topic (optional):** Experiment tracking with MLflow inside the DVC pipeline

**Decision Participants:** All team members

**Decision:** Integrate DVC pipelines with MLFlow to combine the reproducibility provided by DVC with the easy experiment visualization and more advanced features provided by MLFlow. DVC key features allow for easy reproducibility of the pipeline and can also be used for experiment tracking, but lacks the visualization plots and out-of-the-box features that MLFLow offers.

To avoid data redundancy, all preprocessing data is tracked in DVC (necessary in order to benefit form caching stages outputs) including the trained models files. Currently, only light-weight data is being logged in MLFLow, including model parameters, model cross-validation scores,test metrics, carbon emissions produced while training and a reference to the git ref of the DVC experiment which allows to restore exactly the same code and data version used in the process.

Experiments need to be conducted using the `dvc exp run -n <experiment-name>` command in order to be reproducible. Then, they can be restored using  `dvc exp pull` / `dvc exp apply`.


**Rationale:**
The pipeline includes stages that evaluate based on the training results. This obliges to track training data, such as models or metrics in DVC. Nevertheless, we wanted to also benefit from easy visualization charts of MLFlow and possible functionalities regarding model serving in the future. The idea was to use MLFlow as the main tool for developers to assess their experiments and DVC to be able to reproduce them. So, a way of linking both tools needed to be addressed.

A first approach was to log the same information about DVC in MLFlow, but this was rejected since it implied duplicating all data. Instead, we aimed to log to MLFLow only the metrics and useful information that was required to be visualized. To allow experiment reproducibility two options were rejected for the following reasons:
- **Logging current code state and data as an artifact**: This allows reproducing every experiment, but implies manually storing a snapshot of the workspàce in every experiment, which can grow quickly in required storage.
- **Logging the latest commit sha**: This is a common practice, but we consider it does not satisfy our needs. MLFlow is supposed to solve the problem of forgetting the conditions on which a developer has obtained an specific result. However, the common workflow of a developer consists on not committing results until they are sure they have the best result. In the case of machine learning experiments, this usually implies that many experiments are run before finally committing the best one. In this scenario, if we only logged to MLFlow based on the committed changes (e.g. in a CI/CD pipeline), the developer still needs to remember how to reproduce an uncommited experiment. If we logged every run to MLFlow (even if it is not commited) instead, we would not be considering the uncommitted changes in the code or dat (notice this problem is what the previous option solved).

Ater investigating, we found that the command `dvc exp run` does exactly that: It creates a git ref containing a snapshot of the current workspace state, and this can be pushed to dvc. Then, for each run, we log the experiment name as a tag in MLFlow. This way, and always considering that the experiment was pushed, everybody can reproduce an experiment by pulling it using the name logged in MLFLow. Notice there is an implicit mandatory rule: For experiments to be reproducible, they need to be run using `dvc exp run`. If the training script is run manually, the result will be logged to MLFlow, but there will not be any artifact in DVC that allows to restore the workspace state.



**AI Involvement:**

[ ] No AI involvement

[x] Information seeking

[ ] Alternative generation

[x] Alternative assessment

[ ] Recommendation

[x] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Response to AI:** If AI was involved, select one.

[ ] Accepted

[x] Accepted with modifications

[ ] Rejected

[x] Used as input for further analysis

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Assessment of the AI Contribution:** The AI proposed and implemented the run hierarchy, the tag-based parent lookup, the restore tags and the selection stage. It also checked DVC internals (the `DVC_EXP_NAME` env var, the exp ref format, list interpolation in `cmd`) against the installed versions. Modifications came from the team:

* Logging datasets with `log_input` instead of MD5 params was raised by the team member and then adopted.
* Keeping all metric outputs in the DVC cache, instead of git-tracked, was the team member's choice.
* Following EDN-004 and the new module naming was required when merging `dev`.