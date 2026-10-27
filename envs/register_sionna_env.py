"""Configure authorized private data without embedding machine-specific paths."""
import os
from pathlib import Path
from gymnasium.envs.registration import register
from envs.sionna_grid_env import SionnaGrid2DEnv


def make_sionna_grid2d_env(render_mode=None, debug=False, **env_kwargs):
    defaults = dict(
        npz_path=os.environ.get("RF_GRID24_PATH", ""),
        max_steps=20,
        progress_scale=1.0,
        success_bonus=5.0,
        step_cost=0.01,
        debug=debug,
        render_mode=render_mode,
    )
    defaults.update(grid96_npz_path=os.environ.get("RF_GRID96_PATH", ""), rep_mode="24")
    defaults.update(env_kwargs)
    if not defaults["npz_path"] or not Path(defaults["npz_path"]).is_file():
        raise FileNotFoundError("Set RF_GRID24_PATH to an authorized private data archive; no data is distributed.")
    if defaults.get("rep_mode") == "96" and (not defaults.get("grid96_npz_path") or not Path(defaults["grid96_npz_path"]).is_file()):
        raise FileNotFoundError("Set RF_GRID96_PATH or pass grid96_npz_path for authorized private data.")
    return SionnaGrid2DEnv(**defaults)


register(id="SionnaGrid2D-v0", entry_point="envs.register_sionna_env:make_sionna_grid2d_env")
