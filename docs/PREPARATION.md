# Repository notes

This repository was prepared on October 27, 2026 from the research code archive. It contains selected source files and an archived environment specification.

## Source organization

- Two PPO training scripts, a 2D evaluator, and two environment implementations.
- The original top-level `envs` and `agents` import structure.
- Environment variables for dataset paths, replacing machine-specific defaults.
- Explicit errors when required private data files are missing.
- A release allowlist and checker for public repository contents.

The learning and feature-extraction algorithms retain their archived implementations.

## Data and outputs

The public release excludes raw and processed measurements, dataset construction scripts, sample arrays, fitted normalization values, checkpoints, logs, trajectories, videos, figures, and paper assets. Authorized compatible data must be supplied separately using the documented interface.

## Validation and known limitations

Syntax and release-content checks are available through `python tools/check_release.py`. Training, inference, dependency installation, and reproduction of the paper's results have not been performed for this release.

The 3D trainer retains archived 2D-specific trajectory utilities. Use `trace_every=0` for 3D experiments. The evaluator covers 2D and supports a subset of the training configurations.

## Attribution

CleanRL licensing is retained separately. No additional license is granted for the research author's contributions. The [third-party notices](../THIRD_PARTY_NOTICES.md) identify upstream material.
