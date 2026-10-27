# Local preparation record

Prepared October 27, 2026. Originals were not modified. Only a fixed list of text source files and the archived environment specification were read into this draft. No datasets or generated artifacts were copied, and no original Git history was imported.

## Changes

- Selected two PPO trainers, one 2D evaluator, and two environment implementations.
- Preserved the existing envs import structure using top-level envs and agents directories.
- Replaced registration defaults with environment-variable-based private paths and explicit missing-file errors.
- Replaced the trainer's hard-coded optional data path with RF_GRID96_PATH.
- Added documentation, a default-deny Git ignore file, an explicit release allowlist, and a release checker.
- Preserved upstream CleanRL licensing separately. No blanket license was assigned to the research author's contributions.

## Deliberate exclusions

All raw/processed research data, data loaders and grid-building scripts, sample arrays, fitted normalization values, checkpoints, logs, runs, evaluation outputs, videos, trajectories, figures, papers, virtual environments, and embedded Git metadata are excluded.

## Limits of review

The draft contains research source and constant experiment parameters, not copied measurement records. Automated checks are a useful screen, not proof that source code is institutionally approved for disclosure. The author approved a separate public code-only companion on October 27, 2026. It contains the selected text source and documentation only; confidential research data and generated artifacts are not distributed. The private repository remains unchanged.

No training or inference was run, and no dependency environment was installed. Do not interpret syntax checks as runtime or scientific validation. The 3D trainer retains some archived 2D-specific trajectory utilities; leave trace_every at zero for 3D. The 2D evaluator does not cover every feature configuration or the 3D environment.
