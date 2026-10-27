"""Configure authorized private data without embedding machine-specific paths."""
import os
from pathlib import Path
from gymnasium.envs.registration import register
from envs.sionna_grid3d_env import SionnaGrid3DEnv


def make_sionna_grid3d_env(render_mode=None, debug=False, **env_kwargs):
    defaults = dict(
        npz_path=os.environ.get("RF_GRID3D_PATH", ""),
        max_steps=40,
        progress_scale=1.0,
        success_bonus=10.0,
        step_cost=0.01,
        debug=debug,
        render_mode=render_mode,
    )
    defaults.update(env_kwargs)
    if not defaults["npz_path"] or not Path(defaults["npz_path"]).is_file():
        raise FileNotFoundError("Set RF_GRID3D_PATH to an authorized private data archive; no data is distributed.")
    return SionnaGrid3DEnv(**defaults)


register(id="SionnaGrid3D-v0", entry_point="envs.register_sionna_grid3d:make_sionna_grid3d_env")
