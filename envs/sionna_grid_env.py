import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SionnaGrid2DEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        npz_path: str,
        max_steps: int = 20,
        progress_scale: float = 1.0,
        success_bonus: float = 5.0,
        step_cost: float = 0.01,
        debug: bool = False,
        render_mode: str | None = None,
        use_observation: bool = True,
        goal_only: bool = False,
        obs_mode: str = "raw",
        feature_noise_std: float = 0.0,
        rep_mode: str = "24",
        grid96_npz_path: str | None = None,
        stat_index: int = 11,
    ):
        super().__init__()

        data = np.load(npz_path, allow_pickle=False)

        # Stored representatives
        self.grid_obs = data["grid_obs"]      # (24, 4, 1024)
        self.grid_pos = data["grid_pos"]
        self.grid_rc = data["grid_rc"]
        self.grid_az = data["grid_az"]
        self.grid_el = data["grid_el"]
        self.grid_stats = data["grid_stats"].astype(np.float32)   # (24,88)

        # Saved normalization stats from the selected z-plane
        self.real_mean = data["real_mean"].astype(np.float32)
        self.real_std = data["real_std"].astype(np.float32)
        self.imag_mean = data["imag_mean"].astype(np.float32)
        self.imag_std = data["imag_std"].astype(np.float32)
        self.rms_mean = data["rms_mean"].astype(np.float32)   # (4,)
        self.rms_std = data["rms_std"].astype(np.float32)     # (4,)
        

        self.n_cells = int(self.grid_obs.shape[0])
        assert self.n_cells == 24, f"Expected 24 cells, got {self.n_cells}"
        assert self.grid_obs.shape[1:] == (4, 1024), f"Expected (24,4,1024), got {self.grid_obs.shape}"
        assert self.grid_rc.shape == (24, 2), f"Expected grid_rc shape (24,2), got {self.grid_rc.shape}"
        assert self.grid_stats.shape == (24, 88), f"Expected grid_stats shape (24,88), got {self.grid_stats.shape}"
        assert self.rms_mean.shape == (4,), f"Expected rms_mean shape (4,), got {self.rms_mean.shape}"
        assert self.rms_std.shape == (4,), f"Expected rms_std shape (4,), got {self.rms_std.shape}"
        

        self.n_rows = 4
        self.n_cols = 6

        self.max_steps = int(max_steps)
        self.progress_scale = float(progress_scale)
        self.success_bonus = float(success_bonus)
        self.step_cost = float(step_cost)
        self.debug = bool(debug)
        self.render_mode = render_mode

        # Observation ablation flags
        self.use_observation = bool(use_observation)
        self.goal_only = bool(goal_only)
        self.feature_noise_std = float(feature_noise_std)
        assert self.feature_noise_std >= 0.0, "feature_noise_std must be >= 0"
        self.rep_mode = str(rep_mode)
        assert self.rep_mode in ["24", "96"], f"Unsupported rep_mode: {self.rep_mode}"

        self.grid96_npz_path = grid96_npz_path
        self.use_grid96 = (self.rep_mode == "96")
        self.stat_index = int(stat_index)
        assert 0 <= self.stat_index < 22, f"stat_index must be in [0,21], got {self.stat_index}"
        
        self.obs_mode = str(obs_mode)
        assert self.obs_mode in ["raw", "rms", "coords", "synthetic_gradient", "stats", "rms_phase", "phase_only", "single_stat", "iq_mean", "iq_std"], f"Unsupported obs_mode: {self.obs_mode}"
       
        if self.use_grid96:
            if self.grid96_npz_path is None:
                raise ValueError("grid96_npz_path must be provided when rep_mode='96'")

            data96 = np.load(self.grid96_npz_path, allow_pickle=False)

            grid_obs_96_flat = data96["grid_obs_96"]        # (96,4,1024)
            grid_pos_96_flat = data96["grid_pos_96"]        # (96,3)
            grid_rc_96_flat = data96["grid_rc_96"]          # (96,2)
            grid_sector_96_flat = data96["grid_sector_96"]  # (96,)
            grid_az_96_flat = data96["grid_az_96"]          # (96,2)
            grid_el_96_flat = data96["grid_el_96"]          # (96,2)

            self.grid_stats_96_flat = None
            if "grid_stats_96" in data96:
                self.grid_stats_96_flat = data96["grid_stats_96"].astype(np.float32)   # (96,88)

            assert grid_obs_96_flat.shape == (96, 4, 1024), f"Expected (96,4,1024), got {grid_obs_96_flat.shape}"
            assert grid_pos_96_flat.shape == (96, 3), f"Expected (96,3), got {grid_pos_96_flat.shape}"
            assert grid_rc_96_flat.shape == (96, 2), f"Expected (96,2), got {grid_rc_96_flat.shape}"
            assert grid_sector_96_flat.shape == (96,), f"Expected (96,), got {grid_sector_96_flat.shape}"

            self.grid_obs_96 = grid_obs_96_flat.reshape(24, 4, 4, 1024)
            self.grid_pos_96 = grid_pos_96_flat.reshape(24, 4, 3)
            self.grid_rc_96 = grid_rc_96_flat.reshape(24, 4, 2)
            self.grid_sector_96 = grid_sector_96_flat.reshape(24, 4)
            self.grid_az_96 = grid_az_96_flat.reshape(24, 4, 2)
            self.grid_el_96 = grid_el_96_flat.reshape(24, 4, 2)

            if self.grid_stats_96_flat is not None:
                self.grid_stats_96 = self.grid_stats_96_flat.reshape(24, 4, 88)
            else:
                self.grid_stats_96 = None
        else:
            self.grid_obs_96 = None
            self.grid_pos_96 = None
            self.grid_rc_96 = None
            self.grid_sector_96 = None
            self.grid_az_96 = None
            self.grid_el_96 = None
            self.grid_stats_96 = None
            
        self.rc_to_idx = {}
        self.idx_to_rc = {}
        for idx, (r, c) in enumerate(self.grid_rc):
            r = int(r)
            c = int(c)
            self.rc_to_idx[(r, c)] = int(idx)
            self.idx_to_rc[int(idx)] = (r, c)
            
        self.synthetic_gradient_values = self._build_synthetic_gradient_values()
        self.rms_phase_mean, self.rms_phase_std = self._compute_rms_phase_norm_stats_from_grid_obs()
        self.phase_mean, self.phase_std = self._compute_phase_norm_stats_from_grid_obs()
        self.single_stat_mean, self.single_stat_std = self._compute_single_stat_norm_stats_from_grid_obs(self.stat_index)
        self.iq_mean_feat_mean, self.iq_mean_feat_std = self._compute_iq_mean_norm_stats_from_grid_obs()
        self.iq_std_feat_mean, self.iq_std_feat_std = self._compute_iq_std_norm_stats_from_grid_obs()

        self.action_space = spaces.Discrete(4)

        if self.obs_mode == "raw":
            obs_shape = (16, 1024)
        elif self.obs_mode == "rms":
            obs_shape = (8,)
        elif self.obs_mode == "coords":
            obs_shape = (4,)
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
        
        self.cur_rep_id: int | None = None
        self.goal_rep_id: int | None = None

    def _normalize_complex_obs(self, x_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x_complex: (4,1024) complex64
        returns:
            xr_norm: (4,1024) float32
            xi_norm: (4,1024) float32
        """
        xr = x_complex.real.astype(np.float32)
        xi = x_complex.imag.astype(np.float32)

        xr = (xr - self.real_mean[:, None]) / (self.real_std[:, None] + 1e-8)
        xi = (xi - self.imag_mean[:, None]) / (self.imag_std[:, None] + 1e-8)

        return xr, xi

    def _complex4_to_real8_normalized(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (8,1024) float32
        """
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
        """
        Returns:
        values: (24,) float32
        A simple center-peaked scalar field over the 4x6 grid, normalized to [0,1].
        """
        center_r = (self.n_rows - 1) / 2.0   # 1.5
        center_c = (self.n_cols - 1) / 2.0   # 2.5

        vals = np.zeros(self.n_cells, dtype=np.float32)

        for idx in range(self.n_cells):
            r, c = self.idx_to_rc[int(idx)]
            d = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
            vals[idx] = -d

        # normalize to [0,1]
        vmin = vals.min()
        vmax = vals.max()
        vals = (vals - vmin) / (vmax - vmin + 1e-8)

        return vals.astype(np.float32)
    
    def _complex4_to_iq_mean8(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (8,) float32
        [mean(Re1), mean(Im1), mean(Re2), mean(Im2), ...]
        """
        xr = x_complex.real.astype(np.float32)   # (4,1024)
        xi = x_complex.imag.astype(np.float32)   # (4,1024)

        mean_r = np.mean(xr, axis=1)   # (4,)
        mean_i = np.mean(xi, axis=1)   # (4,)

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
        """
        x_complex: (4,1024) complex64
        returns:   (8,) float32
        [std(Re1), std(Im1), std(Re2), std(Im2), ...]
        """
        xr = x_complex.real.astype(np.float32)   # (4,1024)
        xi = x_complex.imag.astype(np.float32)   # (4,1024)

        std_r = np.std(xr, axis=1)   # (4,)
        std_i = np.std(xi, axis=1)   # (4,)

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
    
    def _sample_rep_id_for_cell(self, cell_idx: int) -> int:
        """
        For the 96-rep setting, sample one of 4 representatives for a coarse cell.
        For the 24-rep setting, always return 0.
        """
        if not self.use_grid96:
            return 0
        return int(self.np_random.integers(0, 4))
    
    def _get_single_stat4_for_cell_rep(self, cell_idx: int, rep_id: int, stat_index: int) -> np.ndarray:
        """
        Returns one selected handcrafted stat across the 4 channels.
        Output shape: (4,)
        """
        stats_flat = self._get_stats_for_cell_rep(cell_idx, rep_id)   # (88,)
        stats_4x22 = stats_flat.reshape(4, 22)                        # (4,22)
        feat4 = stats_4x22[:, int(stat_index)]                        # (4,)
        return feat4.astype(np.float32)
    
    def _compute_iq_mean_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []

        if self.use_grid96:
            for cell_idx in range(24):
                for rep_id in range(4):
                    feat = self._complex4_to_iq_mean8(self._get_complex_obs_for_cell_rep(cell_idx, rep_id))
                    feats.append(feat)
        else:
            for cell_idx in range(self.grid_obs.shape[0]):
                feat = self._complex4_to_iq_mean8(self.grid_obs[cell_idx])
                feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = np.maximum(feats.std(axis=0), 1e-8).astype(np.float32)
        return feat_mean.astype(np.float32), feat_std.astype(np.float32)


    def _compute_iq_std_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []

        if self.use_grid96:
            for cell_idx in range(24):
                for rep_id in range(4):
                    feat = self._complex4_to_iq_std8(self._get_complex_obs_for_cell_rep(cell_idx, rep_id))
                    feats.append(feat)
        else:
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
        """
        Compute mean/std for one selected handcrafted stat across the 4 channels.
        Returns:
            feat_mean: (4,)
            feat_std:  (4,)
        """
        feats = []

        if self.use_grid96:
            for cell_idx in range(24):
                for rep_id in range(4):
                    feat4 = self._get_single_stat4_for_cell_rep(cell_idx, rep_id, stat_index)
                    feats.append(feat4)
        else:
            for cell_idx in range(self.grid_obs.shape[0]):
                feat4 = self._get_single_stat4_for_cell_rep(cell_idx, 0, stat_index)
                feats.append(feat4)

        feats = np.stack(feats, axis=0).astype(np.float32)   # (N,4)
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0)
        feat_std = np.maximum(feat_std, 1e-8).astype(np.float32)

        return feat_mean.astype(np.float32), feat_std.astype(np.float32)

    def _obs_from_indices(self, cur_idx: int, goal_idx: int, cur_rep_id: int, goal_rep_id: int) -> np.ndarray:
        """
        Modes:
        - raw: current raw RF + goal raw RF -> (16,1024)
        - rms: current RMS + goal RMS -> (8,)
        - coords: [cur_r, cur_c, goal_r, goal_c] normalized -> (4,)
        - synthetic_gradient: [g(cur_cell), g(goal_cell)] -> (2,)
        - blind / goal-only behavior still controlled by use_observation / goal_only
        """
        if self.obs_mode == "raw":
            if not self.use_observation:
                return np.zeros((16, 1024), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_obs = self._complex4_to_real8_normalized(cur_complex)
            goal_obs = self._complex4_to_real8_normalized(goal_complex)

            if self.goal_only:
                cur_obs = np.zeros_like(cur_obs)

            return np.concatenate([cur_obs, goal_obs], axis=0).astype(np.float32)

        elif self.obs_mode == "rms":
            if not self.use_observation:
                return np.zeros((8,), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_rms = self._complex4_to_rms4(cur_complex)
            goal_rms = self._complex4_to_rms4(goal_complex)

            cur_rms = self._normalize_rms4(cur_rms)
            goal_rms = self._normalize_rms4(goal_rms)

            if self.goal_only:
                cur_rms = np.zeros_like(cur_rms)

            obs = np.concatenate([cur_rms, goal_rms], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "coords":
            if not self.use_observation:
                return np.zeros((4,), dtype=np.float32)

            cur_r, cur_c = self.idx_to_rc[int(cur_idx)]
            goal_r, goal_c = self.idx_to_rc[int(goal_idx)]

            cur_coords = np.array([
                cur_r / (self.n_rows - 1),
                cur_c / (self.n_cols - 1),
            ], dtype=np.float32)

            goal_coords = np.array([
                goal_r / (self.n_rows - 1),
                goal_c / (self.n_cols - 1),
            ], dtype=np.float32)

            if self.goal_only:
                cur_coords = np.zeros_like(cur_coords)

            obs = np.concatenate([cur_coords, goal_coords], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs

        elif self.obs_mode == "synthetic_gradient":
            if not self.use_observation:
                return np.zeros((2,), dtype=np.float32)

            cur_g = np.array([self.synthetic_gradient_values[int(cur_idx)]], dtype=np.float32)
            goal_g = np.array([self.synthetic_gradient_values[int(goal_idx)]], dtype=np.float32)

            if self.goal_only:
                cur_g = np.zeros_like(cur_g)

            obs = np.concatenate([cur_g, goal_g], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "stats":
            if not self.use_observation:
                return np.zeros((176,), dtype=np.float32)

            cur_stats = self._get_stats_for_cell_rep(cur_idx, cur_rep_id)
            goal_stats = self._get_stats_for_cell_rep(goal_idx, goal_rep_id)

            if self.goal_only:
                cur_stats = np.zeros_like(cur_stats)

            obs = np.concatenate([cur_stats, goal_stats], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "rms_phase":
            if not self.use_observation:
                return np.zeros((32,), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_feat = self._complex4_to_rms_phase16(cur_complex)
            goal_feat = self._complex4_to_rms_phase16(goal_complex)

            cur_feat = self._normalize_rms_phase16(cur_feat)
            goal_feat = self._normalize_rms_phase16(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "phase_only":
            if not self.use_observation:
                return np.zeros((24,), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_feat = self._complex4_to_phase12(cur_complex)
            goal_feat = self._complex4_to_phase12(goal_complex)

            cur_feat = self._normalize_phase12(cur_feat)
            goal_feat = self._normalize_phase12(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "single_stat":
            if not self.use_observation:
                return np.zeros((8,), dtype=np.float32)

            cur_feat = self._get_single_stat4_for_cell_rep(cur_idx, cur_rep_id, self.stat_index)    # (4,)
            goal_feat = self._get_single_stat4_for_cell_rep(goal_idx, goal_rep_id, self.stat_index)  # (4,)

            cur_feat = self._normalize_single_stat4(cur_feat)
            goal_feat = self._normalize_single_stat4(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs
        
        elif self.obs_mode == "iq_mean":
            if not self.use_observation:
                return np.zeros((16,), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_feat = self._complex4_to_iq_mean8(cur_complex)
            goal_feat = self._complex4_to_iq_mean8(goal_complex)

            cur_feat = self._normalize_iq_mean8(cur_feat)
            goal_feat = self._normalize_iq_mean8(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs

        elif self.obs_mode == "iq_std":
            if not self.use_observation:
                return np.zeros((16,), dtype=np.float32)

            cur_complex = self._get_complex_obs_for_cell_rep(cur_idx, cur_rep_id)
            goal_complex = self._get_complex_obs_for_cell_rep(goal_idx, goal_rep_id)

            cur_feat = self._complex4_to_iq_std8(cur_complex)
            goal_feat = self._complex4_to_iq_std8(goal_complex)

            cur_feat = self._normalize_iq_std8(cur_feat)
            goal_feat = self._normalize_iq_std8(goal_feat)

            if self.goal_only:
                cur_feat = np.zeros_like(cur_feat)

            obs = np.concatenate([cur_feat, goal_feat], axis=0).astype(np.float32)
            obs = self._maybe_add_feature_noise(obs)
            return obs

        else:
            raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")
        
    def _normalize_single_stat4(self, feat4: np.ndarray) -> np.ndarray:
        return ((feat4 - self.single_stat_mean) / (self.single_stat_std + 1e-8)).astype(np.float32)
        
    def _normalize_rms4(self, rms4: np.ndarray) -> np.ndarray:
        """
        rms4: (4,) float32
        returns: (4,) float32 normalized RMS per channel
        """
        return ((rms4 - self.rms_mean) / (self.rms_std + 1e-8)).astype(np.float32)
        
    def _manhattan(self, idx_a: int, idx_b: int) -> int:
        ra, ca = self.idx_to_rc[int(idx_a)]
        rb, cb = self.idx_to_rc[int(idx_b)]
        return abs(ra - rb) + abs(ca - cb)

    def _euclidean_xy(self, idx_a: int, idx_b: int) -> float:
        pa = self.grid_pos[int(idx_a), :2]   # x,y
        pb = self.grid_pos[int(idx_b), :2]
        return float(np.linalg.norm(pa - pb))
    
    def _complex4_to_rms4(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (4,) float32 RMS per channel
        """
        power = (x_complex.real.astype(np.float32) ** 2 + x_complex.imag.astype(np.float32) ** 2)
        rms = np.sqrt(np.mean(power, axis=1))
        return rms.astype(np.float32)
    
    def _complex4_to_rms_phase16(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (16,) float32

        Features:
        - RMS per channel: 4 dims
        - For antenna pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3):
            mean phase difference, std phase difference
        => 6 * 2 = 12 dims
        Total = 16 dims
        """
        if x_complex.shape != (4, 1024):
            raise ValueError(f"Expected x_complex shape (4,1024), got {x_complex.shape}")

        # 1) RMS per channel
        power = (x_complex.real.astype(np.float64) ** 2 + x_complex.imag.astype(np.float64) ** 2)
        rms = np.sqrt(np.mean(power, axis=1))   # (4,)

        # 2) Phase-difference features between antenna pairs
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        phase_feats = np.zeros(12, dtype=np.float64)

        for idx, (i, j) in enumerate(pairs):
            phase_diff = np.angle(x_complex[i] * np.conj(x_complex[j]))   # (1024,)
            phase_feats[idx * 2] = np.mean(phase_diff)
            phase_feats[idx * 2 + 1] = np.std(phase_diff)

        feat = np.concatenate([rms, phase_feats], axis=0).astype(np.float32)   # (16,)
        return feat
    
    def _complex4_to_phase12(self, x_complex: np.ndarray) -> np.ndarray:
        """
        x_complex: (4,1024) complex64
        returns:   (12,) float32

        Features:
        For antenna pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3):
        - mean phase difference
        - std phase difference

        Total = 6 * 2 = 12
        """
        if x_complex.shape != (4, 1024):
            raise ValueError(f"Expected x_complex shape (4,1024), got {x_complex.shape}")

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        phase_feats = np.zeros(12, dtype=np.float64)

        for idx, (i, j) in enumerate(pairs):
            phase_diff = np.angle(x_complex[i] * np.conj(x_complex[j]))   # (1024,)
            phase_feats[idx * 2] = np.mean(phase_diff)
            phase_feats[idx * 2 + 1] = np.std(phase_diff)

        return phase_feats.astype(np.float32)
    
    def _compute_rms_phase_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []

        if self.use_grid96:
            for cell_idx in range(24):
                for rep_id in range(4):
                    feat = self._complex4_to_rms_phase16(self.grid_obs_96[cell_idx, rep_id])  # (16,)
                    feats.append(feat)
        else:
            for i in range(self.grid_obs.shape[0]):
                feat = self._complex4_to_rms_phase16(self.grid_obs[i])  # (16,)
                feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0)
        feat_std = np.maximum(feat_std, 1e-8).astype(np.float32)

        return feat_mean.astype(np.float32), feat_std.astype(np.float32)
    
    def _normalize_rms_phase16(self, feat16: np.ndarray) -> np.ndarray:
        """
        feat16: (16,) float32
        returns: normalized (16,) float32
        """
        return ((feat16 - self.rms_phase_mean) / (self.rms_phase_std + 1e-8)).astype(np.float32)
    
    def _compute_phase_norm_stats_from_grid_obs(self) -> tuple[np.ndarray, np.ndarray]:
        feats = []

        if self.use_grid96:
            for cell_idx in range(24):
                for rep_id in range(4):
                    feat = self._complex4_to_phase12(self.grid_obs_96[cell_idx, rep_id])  # (12,)
                    feats.append(feat)
        else:
            for i in range(self.grid_obs.shape[0]):
                feat = self._complex4_to_phase12(self.grid_obs[i])  # (12,)
                feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)
        feat_mean = feats.mean(axis=0)
        feat_std = feats.std(axis=0)
        feat_std = np.maximum(feat_std, 1e-8).astype(np.float32)

        return feat_mean.astype(np.float32), feat_std.astype(np.float32)
    
    def _get_complex_obs_for_cell_rep(self, cell_idx: int, rep_id: int) -> np.ndarray:
        """
        Returns complex observation (4,1024) for a given coarse cell and rep id.
        """
        if self.use_grid96:
            return self.grid_obs_96[int(cell_idx), int(rep_id)]
        else:
            return self.grid_obs[int(cell_idx)]
        
    def _get_stats_for_cell_rep(self, cell_idx: int, rep_id: int) -> np.ndarray:
        """
        Returns stats vector (88,) for a given coarse cell and rep id.
        """
        if self.use_grid96:
            if self.grid_stats_96 is None:
                raise ValueError("grid_stats_96 not available in the 96-rep npz")
            return self.grid_stats_96[int(cell_idx), int(rep_id)]
        else:
            return self.grid_stats[int(cell_idx)]


    def _normalize_phase12(self, feat12: np.ndarray) -> np.ndarray:
        """
        feat12: (12,) float32
        returns: normalized (12,) float32
        """
        return ((feat12 - self.phase_mean) / (self.phase_std + 1e-8)).astype(np.float32)

    def _move(self, idx: int, action: int) -> int:
        r, c = self.idx_to_rc[int(idx)]

        if action == self.UP:
            r2, c2 = max(0, r - 1), c
        elif action == self.DOWN:
            r2, c2 = min(self.n_rows - 1, r + 1), c
        elif action == self.LEFT:
            r2, c2 = r, max(0, c - 1)
        elif action == self.RIGHT:
            r2, c2 = r, min(self.n_cols - 1, c + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        return self.rc_to_idx[(r2, c2)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cur_idx = int(self.np_random.integers(0, self.n_cells))

        while True:
            goal = int(self.np_random.integers(0, self.n_cells))
            if goal != self.cur_idx:
                self.goal_idx = goal
                break

        self.goal_rep_id = self._sample_rep_id_for_cell(self.goal_idx)
        self.cur_rep_id = self._sample_rep_id_for_cell(self.cur_idx)

        self.prev_dist = float(self._euclidean_xy(self.cur_idx, self.goal_idx))
        self.steps = 0
        self.total_reward = 0.0
        self.trajectory = [self.cur_idx]

        obs = self._obs_from_indices(self.cur_idx, self.goal_idx, self.cur_rep_id, self.goal_rep_id)
        info = {
            "cur_idx": int(self.cur_idx),
            "goal_idx": int(self.goal_idx),
            "cur_rep_id": int(self.cur_rep_id),
            "goal_rep_id": int(self.goal_rep_id),
            "distance": float(self.prev_dist),
        }

        if self.debug:
            print(f"[RESET] start={self.cur_idx}, goal={self.goal_idx}, d0={self.prev_dist}")

        return obs, info

    def step(self, action: int):
        self.steps += 1

        next_idx = self._move(self.cur_idx, int(action))

        d_prev = float(self.prev_dist)
        d_cur = float(self._euclidean_xy(next_idx, self.goal_idx))
        progress = d_prev - d_cur

        reward = self.progress_scale * progress - self.step_cost

        success = (next_idx == self.goal_idx)
        if success:
            reward += self.success_bonus

        self.cur_idx = next_idx
        self.cur_rep_id = self._sample_rep_id_for_cell(self.cur_idx)
        self.prev_dist = d_cur
        self.total_reward += reward
        self.trajectory.append(self.cur_idx)

        terminated = success
        truncated = self.steps >= self.max_steps

        obs = self._obs_from_indices(self.cur_idx, self.goal_idx, self.cur_rep_id, self.goal_rep_id)

        info = {
            "cur_idx": int(self.cur_idx),
            "goal_idx": int(self.goal_idx),
            "cur_rep_id": int(self.cur_rep_id),
            "goal_rep_id": int(self.goal_rep_id),
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
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # draw checkerboard
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=1)
                ax.add_patch(rect)

        # draw goal
        gr, gc = self.idx_to_rc[self.goal_idx]
        goal_rect = plt.Rectangle((gc, gr), 1, 1, color="red", alpha=0.5)
        ax.add_patch(goal_rect)

        # draw agent
        cr, cc = self.idx_to_rc[self.cur_idx]
        agent_rect = plt.Rectangle((cc, cr), 1, 1, color="blue", alpha=0.5)
        ax.add_patch(agent_rect)

        for idx in self.trajectory:
            r, c = self.idx_to_rc[idx]
            ax.plot(c + 0.5, r + 0.5, "ko", markersize=2)

        ax.set_xticks(np.arange(self.n_cols) + 0.5)
        ax.set_yticks(np.arange(self.n_rows) + 0.5)
        ax.set_xticklabels(range(self.n_cols))
        ax.set_yticklabels(range(self.n_rows))
        ax.set_title("SionnaGrid2D")
        ax.tick_params(length=0)

        canvas = FigureCanvas(fig)
        canvas.draw()

        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]

        plt.close(fig)
        return image

    def close(self):
        pass