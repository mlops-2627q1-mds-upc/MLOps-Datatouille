# AGENTS.md — MLOps Spam Classifier

## Context
* Binary text classification: spam vs. not spam.
* Goal: clean code and MLOps practices, not only accuracy.
* Base: Cookiecutter Data Science layout. Keep layout. Do not restructure without approval.

## Sources of truth — read live, do not duplicate
* Paths: read `src/config.py`. Import paths from there. No hardcoded paths.
* Style and lint: read `pyproject.toml`. Follow it.
* Workflow and git: read `CONTRIBUTING.md`. Follow it.
* CI: read `.github/workflows/`. Match CI locally before push.
* Deps: read `requirements.txt` and `pyproject.toml`. Do not assume libs.
* Infra: read `infra/`. Do not assume URIs or services.
* Never copy values from these files into this file. Values change. Read them each run.

## Layout roles — stable, logic will evolve
* `src/`: ingestion, features, config. One concern per module. Separate I/O from logic.
* `src/modeling/`: training, experiment logging, and inference only.
* `src/api/`: serving and feedback.
* `src/monitoring/`: drift and performance checks.
* `notebooks/`: exploration only. No reusable logic.
* `data/`, `models/`, `reports/`: outputs. Never import from them.

## Commands — resolve from source files
* Setup, test, lint, run: check `README.md`, `pyproject.toml`, and CI workflow first.
* Use repo-defined entrypoints (`src` modules as CLIs). Do not invent new entrypoints.
* Before PR: run the same lint and test commands that CI runs.

## Code rules
* Type all params and returns. Use Google-style docstrings with Args, Returns, Raises.
* When reading a file: if typing or docstrings are missing, flag it. List file and function. Do not silently ignore.
* Keep functions small and pure where possible. Separate I/O, preprocessing, training, inference.
* Use repo-standard libs for logging, CLI, progress, paths, and env loading. Check imports in `src/` first.
* Follow lint config from `pyproject.toml`. Fix imports before push.
* Load secrets from `.env`. Never hardcode tokens, URIs, or passwords.

## MLOps rules
* Never commit data or models to git. Track large artifacts with DVC. Check `CONTRIBUTING.md` for flow.
* Log all runs: params, metrics, dataset version, code version.
* Keep training deterministic: set seeds, record data hash.
* Keep API schemas backward-compatible. Add tests for new endpoints.
* Monitoring code must not import training code.

## Git rules — see CONTRIBUTING.md
* GitHub Flow only. `main` is protected and deployable.
* Branch from `main` per issue. Use repo branch pattern.
* Commit in imperative style with issue reference. Squash-and-merge only.
* Never commit or push. Only suggest a commit command. User runs it.

## Commit suggestion — required after code
* After lint + tests pass, output a ready-to-copy commit command.
* Follow `CONTRIBUTING.md`: short imperative subject, blank line, body with what and why, issue ref.
* Form:
  `git add <files> && git commit -m "<subject>" -m "<body>"`
* Keep subject <=50 chars. No period. Body explains change. Add `Closes #<id>` when known.

## Agent workflow
* Before code: read target module, its test, and all Sources of truth above. Check open issue ID.
* Make minimal diffs. Do not refactor unrelated files.
* Do not edit `.github/`, `infra/`, lint config, or DVC remote without explicit request.
* After code: run lint + tests. If fail, fix and rerun. Report commands run and result.
* After finish: suggest a clean-context review via subagent. Do not auto-run. Provide copy prompt.
* If blocked: stop, state missing input, list options. Do not guess schema or params.

## Communication
* Lead with result or answer. No intro, no filler.
* Be concise and precise. Use short phrases and bullets.
* Explain only when it adds value. No boilerplate.
* Helpful, not bloatful. No repetition.
