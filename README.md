# RF Observation Feature Ablations for Goal-Conditioned Navigation

Code companion for the observation-representation experiments associated with *Active Sensing with Meta-Reinforcement Learning for Emitter Localization from RF Observations*.

This repository isolates the feature-comparison component of the research: PPO policies receive current and goal RF representations and navigate a discrete Sionna-derived environment. It complements the [thesis portfolio](https://github.com/shamail-khan/gnss-rl-portfolio).

## Confidential data

**No research data or data-derived artifacts are distributed.** Raw measurements, processed grids, normalization values, learned checkpoints, logs, trajectories, videos, figures, and the paper PDF are excluded. The source reads private data supplied separately by an authorized user. No download link or redistribution permission is implied.

## Included code

- `envs/sionna_grid_env.py`: 24-cell 2D navigation with optional multiple representatives.
- `envs/sionna_grid3d_env.py`: 96-cell 3D navigation.
- `agents/ppo_mlp.py`: PPO with an MLP for vector features or a CNN for sequence observations.
- `agents/ppo_1dcnn.py`: archived raw-observation PPO implementation.
- `agents/eval_mlp_sionna_ppo.py`: archived 2D evaluator for matching checkpoints.
- `tools/check_release.py`: file allowlist, text, syntax, size, and credential-pattern checks.

## Experiments and scope

Supported observation modes include raw IQ, IQ mean, IQ standard deviation, RMS, inter-antenna phase statistics, combined RMS/phase, and individual statistics. Coordinate and synthetic-gradient inputs are diagnostic controls, not RF-only methods.

The environment includes the goal RF representation in the policy input. This is goal-conditioned navigation; it does not demonstrate localization without a goal reference.

The 2D and 3D environments differ in actions, episode budgets, and distance shaping. Keep the environment, representative mode, reward, training budget, evaluation protocol, and seeds fixed when comparing features. Raw sequence and compact vector inputs use different encoders, so that comparison also changes architecture. These scripts are not a complete reproduction package for every result in the paper.

## Setup with authorized private data

The archived `environment.yml` is provided as a starting specification, not a validated installation. Python 3.10 is specified. Install in a separate environment.

Set local environment variables to private files outside this repository:
- `RF_GRID24_PATH`: 24-cell processed grid.
- `RF_GRID96_PATH`: optional 96 representatives for the same 2D coarse grid.
- `RF_GRID3D_PATH`: 96-cell 3D processed grid.

No real files or numeric dataset values are included. See [data interface](docs/DATA_INTERFACE.md) for the source-level contract.

From this directory, an illustrative 2D run is:

```sh
python agents/ppo_mlp.py --env-id SionnaGrid2D-v0 --obs-mode phase_only --rep-mode 24 --total-timesteps 200000 --seed 1
```

The command is an example, not a recovered configuration for a specific paper figure. Leave tracking disabled and do not enable video or trace capture for a public release. Training creates private local artifacts; the release allowlist excludes new files by default.

## Validation and provenance

Repository preparation changed file organization and machine-specific path configuration only. The learning and feature-extraction algorithms were not redesigned. Python syntax and release contents can be checked without opening any dataset:

```sh
python tools/check_release.py
```

Training, inference, dependency installation, and reproduction of paper results have not been performed for this archive. The archived evaluator is for 2D and does not expose all training settings. The latest 3D implementation is included as related experimental code; it has not been mapped to a paper figure.

See [preparation notes](docs/PREPARATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md). This public code-only companion is curated from a separate private code archive. Confidential research data and data-derived artifacts are not distributed.
