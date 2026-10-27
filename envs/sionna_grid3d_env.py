import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SionnaGrid3DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    Z_UP = 4
    Z_DOWN = 5

    def __init__(
        self,
        npz_path: str,
        max_steps: int = 40,
        progress_scale: float = 1.0,
        success_bonus: float = 10.0,
        step_cost: float = 0.01,
        debug: bool = False,
        render_mode: str | None = None,
        use_observation: bool = True,
        goal_only: bool = False,
        obs_mode: str = "raw",
        feature_noise_std: float = 0.0,
        stat_index: int = 11,
        rep_mode: str | None = None,
        grid96_npz_path: str | None = None,
    ):
        super().__init__()

        data = np.load(npz_path, allow_pickle=False)

        # Stored representatives for 96 true 3D cells
        self.grid_obs = data["grid_obs"]              # (96,4,1024)
        self.grid_pos = data["grid_pos"]              # (96,3)
        self.grid_rck = data["grid_rck"]              # (96,3)
        self.grid_az = data["grid_az"]                # (96,2)
        self.grid_el = data["grid_el"]                # (96,2)
        self.grid_stats = data["grid_stats"].astype(np.float32)  # (96,88)

        self.real_mean = data["real_mean"].astype(np.float32)    # (4,)
        self.real_std = data["real_std"].astype(np.float32)      # (4,)
        self.imag_mean = data["imag_mean"].astype(np.float32)    # (4,)
        self.imag_std = data["imag_std"].astype(np.float32)      # (4,)
        self.rms_mean = data["rms_mean"].astype(np.float32)      # (4,)
        self.rms_std = data["rms_std"].astype(np.float32)        # (4,)
        self.z_centers = data["z_centers"].astype(np.float32)    # (4,)

        self.n_cells = int(self.grid_obs.shape[0])
        assert self.n_cells == 96, f"Expected 96 3D cells, got {self.n_cells}"
        assert self.grid_obs.shape[1:] == (4, 1024), f"Expected (96,4,1024), got {self.grid_obs.shape}"
        assert self.grid_rck.shape == (96, 3), f"Expected grid_rck shape (96,3), got {self.grid_rck.shape}"
        assert self.grid_stats.shape == (96, 88), f"Expected grid_stats shape (96,88), got {self.grid_stats.shape}"
        assert self.rms_mean.shape == (4,), f"Expected rms_mean shape (4,), got {self.rms_mean.shape}"
        assert self.rms_std.shape == (4,), f"Expected rms_std shape (4,), got {self.rms_std.shape}"
        assert self.z_centers.shape == (4,), f"Expected z_centers shape (4,), got {self.z_centers.shape}"

        self.n_rows = 4
        self.n_cols = 6
        self.n_layers = 4

        self.max_steps = int(max_steps)
        self.progress_scale = float(progress_scale)
        self.success_bonus = float(success_bonus)
        self.step_cost = float(step_cost)
        self.debug = bool(debug)
        self.render_mode = render_mode

        self.use_observation = bool(use_observation)
        self.goal_only = bool(goal_only)
        self.feature_noise_std = float(feature_noise_std)
        assert self.feature_noise_std >= 0.0, "feature_noise_std must be >= 0"

        self.obs_mode = str(obs_mode)
        assert self.obs_mode in [
            "raw", "rms", "coords", "synthetic_gradient", "stats",
            "rms_phase", "phase_only", "single_stat", "iq_mean", "iq_std",
            "rms_phase_amp_power"
        ], f"Unsupported obs_mode: {self.obs_mode}"

        self.stat_index = int(stat_index)
        assert 0 <= self.stat_index < 22, f"stat_index must be in [0,21], got {self.stat_index}"

        self.rck_to_idx = {}
        self.idx_to_rck = {}
        for idx, (r, c, k) in enumerate(self.grid_rck):
            r = int(r)
            c = int(c)
            k = int(k)
            self.rck_to_idx[(r, c, k)] = int(idx)
            self.idx_to_rck[int(idx)] = (r, c, k)

        self.synthetic_gradient_values = self._build_synthetic_gradient_values()
        self.rms_phase_mean, self.rms_phase_std = self._compute_rms_phase_norm_stats_from_grid_obs()
        self.phase_mean, self.phase_std = self._compute_phase_norm_stats_from_grid_obs()
        self.single_stat_mean, self.single_stat_std = self._compute_single_stat_norm_stats_from_grid_obs(self.stat_index)
        self.iq_mean_feat_mean, self.iq_mean_feat_std = self._compute_iq_mean_norm_stats_from_grid_obs()
        self.iq_std_feat_mean, self.iq_std_feat_std = self._compute_iq_std_norm_stats_from_grid_obs()
        self.rms_phase_amp_power_mean, self.rms_phase_amp_power_std = self._compute_rms_phase_amp_power_norm_stats_from_grid_obs()

        self.action_space = spaces.Discrete(6)

        if self.obs_mode == "raw":
            obs_shape = (16, 1024)
        elif self.obs_mode == "rms":
            obs_shape = (8,)
        elif self.obs_mode == "coords":
            obs_shape = (6,)
        elif self.obs_mode == "synthetic_gradient":
            obs_shape = (2,)
        elif self.obs_mode == "stats":
            obs_shape = (176,)
        elif self.obs_mode == "rms_phase":
            obs_shape = (32,)
        elif self.obs_mode == "phase_only":
            obs_shape = (24,)
        elif self.obs_mode == "single_stat":
            obs_shape = (8,)
        elif self.obs_mode == "iq_mean":
            obs_shape = (16,)
        elif self.obs_mode == "iq_std":
            obs_shape = (16,)
        elif self.obs_mode == "rms_phase_amp_power":
            obs_shape = (48,)
        else:
            raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

        self.cur_idx: int | None = None
        self.goal_idx: int | None = None
        self.prev_dist: float = 0.0
        self.steps: int = 0
        self.total_reward: float = 0.0
        self.trajectory: list[int] = []

    def _normalize_complex_obs(self, x_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xr = x_complex.real.astype(np.float32)
        xi = x_complex.imag.astype(np.float32)

        xr = (xr - self.real_mean[:, None]) / (self.real_std[:, None] + 1e-8)
        xi = (xi - self.imag_mean[:, None]) / (self.imag_std[:, None] + 1e-8)

        return xr, xi

    def _complex4_to_real8_normalized(self, x_complex: np.ndarray) -> np.ndarray:
        xr, xi = self._normalize_complex_obs(x_complex)

        out = np.empty((8, x_complex.shape[1]), dtype=np.float32)
        out[0] = xr[0]
        out[1] = xi[0]
        out[2] = xr[1]
        out[3] = xi[1]
        out[4] = xr[2]
        out[5] = xi[2]
        out[6] = xr[3]
        out[7] = xi[3]
        return out

    def _build_synthetic_gradient_values(self) -> np.ndarray:
        center_r = (self.n_rows - 1) / 2.0
        center_c = (self.n_cols - 1) / 2.0
        center_k = (self.n_layers - 1) / 2.0

        vals = np.zeros(self.n_cells, dtype=np.float32)
        for idx in range(self.n_cells):
            r, c, k = self.idx_to_rck[int(idx)]
            d = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2 + (k - center_k) ** 2)
            vals[idx] = -d

        vmin = vals.min()
        vmax = vals.max()
        vals = (vals - vmin) / (vmax - vmin + 1e-8)
        return vals.astype(np.float32)

    def _complex4_to_iq_mean8(self, x_complex: np.ndarray) -> np.ndarray:
        xr = x_complex.real.astype(np.float32)
        xi = x_complex.imag.astype(np.float32)

        mean_r = np.mean(xr, axis=1)
        mean_i = np.mean(xi, axis=1)

        out = np.empty((8,), dtype=np.float32)
        out[0] = mean_r[0]
        out[1] = mean_i[0]
        out[2] = mean_r[1]
        out[3] = mean_i[1]
        out[4] = mean_r[2]
        out[5] = mean_i[2]
        out[6] = mean_r[3]
        out[7] = mean_i[3]
        return out

    def _complex4_to_iq_std8(self, x_complex: np.ndarray) -> np.ndarray:
        xr = x_complex.real.astype(np.float32)
        xi = x_complex.imag.astype(np.float32)

        std_r = np.std(xr, axis=1)
        std_i = np.std(xi, axis=1)

        out = np.empty((8,), dtype=np.float32)
        out[0] = std_r[0]
        out[1] = std_i[0]
        out[2] = std_r[1]
        out[3] = std_i[1]
        out[4] = std_r[2]
        out[5] = std_i[2]
        out[6] = std_r[3]
        out[7] = std_i[3]
        return out
    
    def _complex4_to_rms_phase_amp_power24(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (24,) float32

        Features:
        - RMS per channel: 4
        - Inter-antenna phase stats: 12
        - Mean amplitude per channel: 4
        - Signal power per channel: 4
        """
        if x_complex.shape != (4, 1024):
            raise ValueError(f"Expected x_complex shape (4,1024), got {x_complex.shape}")

        xr = x_complex.real.astype(np.float64)
        xi = x_complex.imag.astype(np.float64)

        power_samples = xr ** 2 + xi ** 2                    # (4,1024)
        rms = np.sqrt(np.mean(power_samples, axis=1))        # (4,)
        signal_power = np.sum(power_samples, axis=1) * 0.1   # (4,)

        amp = np.abs(x_complex).astype(np.float64)
        mean_amp = np.mean(amp, axis=1)                      # (4,)

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        phase_feats = np.zeros(12, dtype=np.float64)

        for idx, (i, j) in enumerate(pairs):
            phase_diff = np.angle(x_complex[i] * np.conj(x_complex[j]))   # (1024,)
            phase_feats[idx * 2] = np.mean(phase_diff)
            phase_feats[idx * 2 + 1] = np.std(phase_diff)

        feat = np.concatenate([rms, phase_feats, mean_amp, signal_power], axis=0)
        return feat.astype(np.float32)
    
    def _compute_rms_phase_amp_power_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for i in range(self.grid_obs.shape[0]):
            feat = self._complex4_to_rms_phase_amp_power24(self.grid_obs[i])
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)


    def _normalize_rms_phase_amp_power24(self, feat24: np.ndarray) -> np.ndarray:
        return ((feat24 - self.rms_phase_amp_power_mean) / (self.rms_phase_amp_power_std + 1e-8)).astype(np.float32)

    def _maybe_add_feature_noise(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        if self.feature_noise_std <= 0.0:
            return obs

        noise = self.np_random.normal(
            loc=0.0,
            scale=self.feature_noise_std,
            size=obs.shape,
        ).astype(np.float32)

        return (obs + noise).astype(np.float32)

    def _get_single_stat4(self, cell_idx: int, stat_index: int) -> np.ndarray:
        stats_flat = self.grid_stats[int(cell_idx)]      # (88,)
        stats_4x22 = stats_flat.reshape(4, 22)          # (4,22)
        feat4 = stats_4x22[:, int(stat_index)]          # (4,)
        return feat4.astype(np.float32)

    def _compute_iq_mean_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for cell_idx in range(self.grid_obs.shape[0]):
            feat = self._complex4_to_iq_mean8(self.grid_obs[cell_idx])
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _compute_iq_std_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for cell_idx in range(self.grid_obs.shape[0]):
            feat = self._complex4_to_iq_std8(self.grid_obs[cell_idx])
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _normalize_iq_mean8(self, feat8: np.ndarray) -> np.ndarray:
        return ((feat8 - self.iq_mean_feat_mean) / (self.iq_mean_feat_std + 1e-8)).astype(np.float32)

    def _normalize_iq_std8(self, feat8: np.ndarray) -> np.ndarray:
        return ((feat8 - self.iq_std_feat_mean) / (self.iq_std_feat_std + 1e-8)).astype(np.float32)

    def _compute_single_stat_norm_stats_from_grid_obs(self, stat_index: int) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for cell_idx in range(self.grid_obs.shape[0]):
            feat4 = self._get_single_stat4(cell_idx, stat_index)
            feats.append(feat4)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feat_std := feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _normalize_single_stat4(self, feat4: np.ndarray) -> np.ndarray:
        return ((feat4 - self.single_stat_mean) / (self.single_stat_std + 1e-8)).astype(np.float32)

    def _get_complex_obs(self, cell_idx: int) -> np.ndarray:
        return self.grid_obs[int(cell_idx)]

    def _get_stats(self, cell_idx: int) -> np.ndarray:
        return self.grid_stats[int(cell_idx)]

    def _complex4_to_rms4(self, x_complex: np.ndarray) -> np.ndarray:
        power = x_complex.real.astype(np.float32) ** 2 + x_complex.imag.astype(np.float32) ** 2
        rms = np.sqrt(np.mean(power, axis=1))
        return rms.astype(np.float32)

    def _normalize_rms4(self, rms4: np.ndarray) -> np.ndarray:
        return ((rms4 - self.rms_mean) / (self.rms_std + 1e-8)).astype(np.float32)

    def _complex4_to_rms_phase16(self, x_complex: np.ndarray) -> np.ndarray:
        if x_complex.shape != (4, 1024):
            raise ValueError(f"Expected x_complex shape (4,1024), got {x_complex.shape}")

        power = x_complex.real.astype(np.float64) ** 2 + x_complex.imag.astype(np.float64) ** 2
        rms = np.sqrt(np.mean(power, axis=1))

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        phase_feats = np.zeros(12, dtype=np.float64)

        for idx, (i, j) in enumerate(pairs):
            phase_diff = np.angle(x_complex[i] * np.conj(x_complex[j]))
            phase_feats[idx * 2] = np.mean(phase_diff)
            phase_feats[idx * 2 + 1] = np.std(phase_diff)

        feat = np.concatenate([rms, phase_feats], axis=0).astype(np.float32)
        return feat

    def _complex4_to_phase12(self, x_complex: np.ndarray) -> np.ndarray:
        if x_complex.shape != (4, 1024):
            raise ValueError(f"Expected x_complex shape (4,1024), got {x_complex.shape}")

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        phase_feats = np.zeros(12, dtype=np.float64)

        for idx, (i, j) in enumerate(pairs):
            phase_diff = np.angle(x_complex[i] * np.conj(x_complex[j]))
            phase_feats[idx * 2] = np.mean(phase_diff)
            phase_feats[idx * 2 + 1] = np.std(phase_diff)

        return phase_feats.astype(np.float32)

    def _compute_rms_phase_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for i in range(self.grid_obs.shape[0]):
            feat = self._complex4_to_rms_phase16(self.grid_obs[i])
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _normalize_rms_phase16(self, feat16: np.ndarray) -> np.ndarray:
        return ((feat16 - self.rms_phase_mean) / (self.rms_phase_std + 1e-8)).astype(np.float32)

    def _compute_phase_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        for i in range(self.grid_obs.shape[0]):
            feat = self._complex4_to_phase12(self.grid_obs[i])
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _normalize_phase12(self, feat12: np.ndarray) -> np.ndarray:
        return ((feat12 - self.phase_mean) / (self.phase_std + 1e-8)).astype(np.float32)

    def _obs_from_indices(self, cur_idx: int, goal_idx: int) -> np.ndarray:
        if self.obs_mode == "raw":
            if not self.use_observation:
                return np.zeros((16, 1024), dtype=np.float32)

            cur_complex = self._get_complex_obs(cur_idx)
            goal_complex = self._get_complex_obs(goal_idx)

            cur_obs = self._complex4_to_real8_normalized(cur_complex)
            goal_obs = self._complex4_to_real8_normalized(goal_complex)

            if self.goal_only:
                cur_obs = np.zeros_like(cur_obs)

            return np.concatenate([cur_obs, goal_obs], axis=0).astype(np.float32)

        elif self.obs_mode == "rms":
            if not self.use_observation:
                return np.zeros((8,), dtype=np.float32)

            cur_rms = self._normalize_rms4(self._complex4_to_rms4(self._get_complex_obs(cur_idx)))
            goal_rms = self._normalize_rms4(self._complex4_to_rms4(self._get_complex_obs(goal_idx)))

            if self.goal_only:
                cur_rms = np.zeros_like(cur_rms)

            obs = np.concatenate([cur_rms, goal_rms], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "coords":
            if not self.use_observation:
                return np.zeros((6,), dtype=np.float32)

            cur_r, cur_c, cur_k = self.idx_to_rck[int(cur_idx)]
            goal_r, goal_c, goal_k = self.idx_to_rck[int(goal_idx)]

            cur_coords = np.array([
                cur_r / (self.n_rows - 1),
                cur_c / (self.n_cols - 1),
                cur_k / (self.n_layers - 1),
            ], dtype=np.float32)

            goal_coords = np.array([
                goal_r / (self.n_rows - 1),
                goal_c / (self.n_cols - 1),
                goal_k / (self.n_layers - 1),
            ], dtype=np.float32)

            if self.goal_only:
                cur_coords = np.zeros_like(cur_coords)

            obs = np.concatenate([cur_coords, goal_coords], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "synthetic_gradient":
            if not self.use_observation:
                return np.zeros((2,), dtype=np.float32)

            cur_g = np.array([self.synthetic_gradient_values[int(cur_idx)]], dtype=np.float32)
            goal_g = np.array([self.synthetic_gradient_values[int(goal_idx)]], dtype=np.float32)

            if self.goal_only:
                cur_g = np.zeros_like(cur_g)

            obs = np.concatenate([cur_g, goal_g], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "stats":
            if not self.use_observation:
                return np.zeros((176,), dtype=np.float32)

            cur_stats = self._get_stats(cur_idx)
            goal_stats = self._get_stats(goal_idx)

            if self.goal_only:
                cur_stats = np.zeros_like(cur_stats)

            obs = np.concatenate([cur_stats, goal_stats], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "rms_phase":
            if not self.use_observation:
                return np.zeros((32,), dtype=np.float32)

            cur_feat = self._normalize_rms_phase16(self._complex4_to_rms_phase16(self._get_complex_obs(cur_idx)))
            goal_feat = self._normalize_rms_phase16(self._complex4_to_rms_phase16(self._get_complex_obs(goal_idx)))

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "phase_only":
            if not self.use_observation:
                return np.zeros((24,), dtype=np.float32)

            cur_feat = self._normalize_phase12(self._complex4_to_phase12(self._get_complex_obs(cur_idx)))
            goal_feat = self._normalize_phase12(self._complex4_to_phase12(self._get_complex_obs(goal_idx)))

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "single_stat":
            if not self.use_observation:
                return np.zeros((8,), dtype=np.float32)

            cur_feat = self._normalize_single_stat4(self._get_single_stat4(cur_idx, self.stat_index))
            goal_feat = self._normalize_single_stat4(self._get_single_stat4(goal_idx, self.stat_index))

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "iq_mean":
            if not self.use_observation:
                return np.zeros((16,), dtype=np.float32)

            cur_feat = self._normalize_iq_mean8(self._complex4_to_iq_mean8(self._get_complex_obs(cur_idx)))
            goal_feat = self._normalize_iq_mean8(self._complex4_to_iq_mean8(self._get_complex_obs(goal_idx)))

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        elif self.obs_mode == "iq_std":
            if not self.use_observation:
                return np.zeros((16,), dtype=np.float32)

            cur_feat = self._normalize_iq_std8(self._complex4_to_iq_std8(self._get_complex_obs(cur_idx)))
            goal_feat = self._normalize_iq_std8(self._complex4_to_iq_std8(self._get_complex_obs(goal_idx)))

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)
        
        elif self.obs_mode == "rms_phase_amp_power":
            if not self.use_observation:
                return np.zeros((48,), dtype=np.float32)

            cur_feat = self._complex4_to_rms_phase_amp_power24(self._get_complex_obs(cur_idx))
            goal_feat = self._complex4_to_rms_phase_amp_power24(self._get_complex_obs(goal_idx))

            cur_feat = self._normalize_rms_phase_amp_power24(cur_feat)
            goal_feat = self._normalize_rms_phase_amp_power24(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            return self._maybe_add_feature_noise(obs)

        else:
            raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")

    def _euclidean_xyz(self, idx_a: int, idx_b: int) -> float:
        pa = self.grid_pos[int(idx_a), :3]
        pb = self.grid_pos[int(idx_b), :3]
        return float(np.linalg.norm(pa - pb))
    
    def _manhattan_grid(self, idx_a: int, idx_b: int) -> int:
        ra, ca, ka = self.idx_to_rck[int(idx_a)]
        rb, cb, kb = self.idx_to_rck[int(idx_b)]
        return abs(ra - rb) + abs(ca - cb) + abs(ka - kb)

    def _move(self, idx: int, action: int) -> int:
        r, c, k = self.idx_to_rck[int(idx)]

        if action == self.UP:
            r2, c2, k2 = max(0, r - 1), c, k
        elif action == self.DOWN:
            r2, c2, k2 = min(self.n_rows - 1, r + 1), c, k
        elif action == self.LEFT:
            r2, c2, k2 = r, max(0, c - 1), k
        elif action == self.RIGHT:
            r2, c2, k2 = r, min(self.n_cols - 1, c + 1), k
        elif action == self.Z_UP:
            r2, c2, k2 = r, c, min(self.n_layers - 1, k + 1)
        elif action == self.Z_DOWN:
            r2, c2, k2 = r, c, max(0, k - 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        return self.rck_to_idx[(r2, c2, k2)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cur_idx = int(self.np_random.integers(0, self.n_cells))
        while True:
            goal = int(self.np_random.integers(0, self.n_cells))
            if goal != self.cur_idx:
                self.goal_idx = goal
                break

        self.prev_dist = float(self._manhattan_grid(self.cur_idx, self.goal_idx))
        self.steps = 0
        self.total_reward = 0.0
        self.trajectory = [self.cur_idx]

        obs = self._obs_from_indices(self.cur_idx, self.goal_idx)
        info = {
            "cur_idx": int(self.cur_idx),
            "goal_idx": int(self.goal_idx),
            "distance": float(self.prev_dist),
        }

        if self.debug:
            print(f"[RESET] start={self.cur_idx}, goal={self.goal_idx}, d0={self.prev_dist:.3f}")

        return obs, info

    def step(self, action: int):
        self.steps += 1

        next_idx = self._move(self.cur_idx, int(action))

        d_prev = float(self.prev_dist)
        d_cur = float(self._manhattan_grid(next_idx, self.goal_idx))
        progress = d_prev - d_cur

        reward = self.progress_scale * progress - self.step_cost

        success = (next_idx == self.goal_idx)
        if success:
            reward += self.success_bonus

        self.cur_idx = next_idx
        self.prev_dist = d_cur
        self.total_reward += reward
        self.trajectory.append(self.cur_idx)

        terminated = success
        truncated = self.steps >= self.max_steps

        obs = self._obs_from_indices(self.cur_idx, self.goal_idx)

        info = {
            "cur_idx": int(self.cur_idx),
            "goal_idx": int(self.goal_idx),
            "distance": float(d_cur),
            "progress": float(progress),
            "success": bool(success),
            "r_total": float(reward),
        }

        if terminated or truncated:
            info["final_distance"] = float(d_cur)
            info["success"] = bool(success)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        cur_r, cur_c, cur_k = self.idx_to_rck[self.cur_idx]
        goal_r, goal_c, goal_k = self.idx_to_rck[self.goal_idx]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=1)
                ax.add_patch(rect)

        if cur_k == goal_k:
            goal_rect = plt.Rectangle((goal_c, goal_r), 1, 1, color="red", alpha=0.5)
            ax.add_patch(goal_rect)

        agent_rect = plt.Rectangle((cur_c, cur_r), 1, 1, color="blue", alpha=0.5)
        ax.add_patch(agent_rect)

        for idx in self.trajectory:
            r, c, k = self.idx_to_rck[idx]
            if k == cur_k:
                ax.plot(c + 0.5, r + 0.5, "ko", markersize=2)

        ax.set_xticks(np.arange(self.n_cols) + 0.5)
        ax.set_yticks(np.arange(self.n_rows) + 0.5)
        ax.set_xticklabels(range(self.n_cols))
        ax.set_yticklabels(range(self.n_rows))
        ax.set_title(
            f"SionnaGrid3D | current z-layer={cur_k} (z≈{self.z_centers[cur_k]:.2f}) | "
            f"goal z-layer={goal_k} (z≈{self.z_centers[goal_k]:.2f})"
        )
        ax.tick_params(length=0)

        canvas = FigureCanvas(fig)
        canvas.draw()

        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]

        plt.close(fig)
        return image

    def close(self):
        pass