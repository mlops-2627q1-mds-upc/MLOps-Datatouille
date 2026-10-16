**Engineering Decision Notebook (EDN) - Entry**

**Milestone:** M1: Project Inception

**Activity / Topic (optional):** Branching strategy, CI/CD, unit and regression testing

**Decision Participants:** All team members

**Decision:** Extend GitHub Flow with a `dev` staging branch and two-tier CI: task branches merge from and to `dev` with fast checks only (Ruff, `tests/unit/`, sample Great Expectations); only `dev` merges to `main` with full validation (full Great Expectations suite, `tests/regression/`). Protect `main` and `dev` with Rulesets and squash-and-merge.

**Rationale:** Plain GitHub Flow (task branch directly to `main`) gives no place to validate new changes before production. Running heavy regression tests and full-dataset GX checks on every task PR is computationally wasteful and unnecessary. This way we can catch issues early in the `dev` branch while keeping fast feedback on task PRs.

**AI Involvement:**

[ ] No AI involvement

[ ] Information seeking

[x] Alternative generation

[ ] Alternative assessment

[ ] Recommendation

[ ] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Response to AI:** If AI was involved, select one.

[ ] Accepted

[ ] Accepted with modifications

[x] Rejected

[ ] Used as input for further analysis

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Assessment of the AI Contribution:** 

AI suggested not using a separate `dev` staging branch and instead merging task branches directly into `main`, which was rejected because it would eliminate the early validation stage and fast feedback loop provided by the `dev` branch.
