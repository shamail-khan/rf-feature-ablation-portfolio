# Private data interface

This document describes array names and shapes read by the source. It contains no observations, fitted normalization values, coordinates, or data-derived statistics.

## 24-cell grid

The 2D environment reads a NumPy archive with:
- grid_obs: complex array shaped (24, 4, 1024).
- grid_pos: per-cell positions, with x and y used for distance.
- grid_rc: row/column indices shaped (24, 2).
- grid_az and grid_el: per-cell angle metadata.
- grid_stats: feature array shaped (24, 88).
- real_mean, real_std, imag_mean, imag_std, rms_mean, rms_std: per-antenna arrays.

These arrays are confidential and must be supplied separately. Source-level schemas do not establish a data release.

## Optional multiple representatives

rep_mode=96 in the 2D environment still uses 24 coarse navigation cells. A second private archive contains grid_obs_96 (96, 4, 1024), grid_pos_96 (96, 3), grid_rc_96 (96, 2), grid_sector_96 (96,), grid_az_96 and grid_el_96 (96, 2), and optionally grid_stats_96 (96, 88).

This is different from the true 3D environment.

## True 3D grid

The 3D environment reads grid_obs (96, 4, 1024), grid_pos (96, 3), grid_rck (96, 3), grid_az and grid_el, grid_stats (96, 88), per-antenna normalization arrays, and z_centers (4,).

## Reproducibility boundary

Preprocessing and original dataset generation are not provided in this draft. A reader without authorized compatible data cannot reproduce results. The environment computes some feature normalization statistics from its loaded representatives; the correct training/evaluation split and leakage controls still need to be established for the intended experiment.

Do not commit private arrays, sample records, actual normalization values, private file locations, or generated outputs.
