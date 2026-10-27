# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from collections import deque
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import envs.register_sionna_env 




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    use_observation: bool = True
    """if False, env returns all-zero observations"""
    goal_only: bool = False
    """if True, current obs is zeroed but goal obs is kept"""
    trace_every: int = 0
    """save a diagnostic episode trace every N training iterations; 0 disables"""
    trace_seed: int = 123
    """fixed seed for the diagnostic rollout"""
    trace_action_mode: str = "sample"
    """how to choose actions in trace rollout: 'sample' or 'greedy'"""

    # Algorithm specific arguments
    env_id: str = "SionnaGrid2D-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, seed=None, use_observation=True, goal_only=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                use_observation=use_observation,
                goal_only=goal_only,
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                env_id,
                use_observation=use_observation,
                goal_only=goal_only,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, feature_dim: int = 64):
        super().__init__()

        obs_shape = envs.single_observation_space.shape
        assert len(obs_shape) == 2, f"Expected (C,L), got {obs_shape}"
        c_in, seq_len = obs_shape
        n_actions = envs.single_action_space.n

        self.encoder = nn.Sequential(
            layer_init(nn.Conv1d(c_in, 32, kernel_size=7, stride=2, padding=3)),
            nn.ReLU(),
            layer_init(nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )   

        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def get_features(self, x):
        x = x.float()   # x: (B,16,1024)
        return self.encoder(x)   # (B,64)

    def get_value(self, x):
        z = self.get_features(x)
        return self.critic(z)

    def get_action_and_value(self, x, action=None):
        z = self.get_features(x)
        logits = self.actor(z)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(z)

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def draw_grid_trajectory(ax, env, cur_idx_arr, goal_idx):
    n_rows = env.unwrapped.n_rows
    n_cols = env.unwrapped.n_cols
    idx_to_rc = env.unwrapped.idx_to_rc

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(n_rows):
        for c in range(n_cols):
            rect = plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=1)
            ax.add_patch(rect)

    gr, gc = idx_to_rc[int(goal_idx)]
    goal_rect = plt.Rectangle((gc, gr), 1, 1, color="red", alpha=0.35)
    ax.add_patch(goal_rect)

    pts = []
    for idx in cur_idx_arr:
        r, c = idx_to_rc[int(idx)]
        pts.append((c + 0.5, r + 0.5))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-o", markersize=4)

    for t, (x, y) in enumerate(pts):
        ax.text(x + 0.03, y - 0.03, str(t), fontsize=8)

    s_idx = int(cur_idx_arr[0])
    sr, sc = idx_to_rc[s_idx]
    start_rect = plt.Rectangle((sc, sr), 1, 1, color="green", alpha=0.25)
    ax.add_patch(start_rect)

    ax.set_title("Grid trajectory")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(range(n_cols))
    ax.set_yticklabels(range(n_rows))
    ax.tick_params(length=0)


def plot_action_probs(ax, action_probs):
    im = ax.imshow(action_probs.T, aspect="auto", origin="lower")
    ax.set_title("Action probabilities")
    ax.set_xlabel("Step")
    ax.set_yticks(range(4))
    ax.set_yticklabels(ACTION_NAMES)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_obs_heatmap(ax, obs_stack):
    T, C, L = obs_stack.shape
    panel = obs_stack.reshape(T * C, L)
    im = ax.imshow(panel, aspect="auto", origin="lower")
    ax.set_title("Observation heatmap (steps stacked)")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Step×channel")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_feature_heatmap(ax, features):
    im = ax.imshow(features, aspect="auto", origin="lower")
    ax.set_title("Encoder features")
    ax.set_xlabel("Feature dim")
    ax.set_ylabel("Step")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_step_table(ax, actions, rewards, distances, cur_idx):
    ax.axis("off")
    lines = ["step | cell | action | reward | dist"]
    max_rows = min(len(actions), 20)
    for t in range(max_rows):
        cell = int(cur_idx[t + 1]) if t + 1 < len(cur_idx) else int(cur_idx[-1])
        lines.append(
            f"{t:>3d} | {cell:>4d} | {ACTION_NAMES[int(actions[t])]:>5s} | "
            f"{rewards[t]:>6.3f} | {distances[t]:>6.3f}"
        )
    if len(actions) > max_rows:
        lines.append("...")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Step summary")


def save_trace_rollout(agent, args, run_name, iteration, device):
    trace_dir = os.path.join("trace_rollouts", run_name)
    os.makedirs(trace_dir, exist_ok=True)

    env = gym.make(
        args.env_id,
        use_observation=args.use_observation,
        goal_only=args.goal_only,
    )

    was_training = agent.training
    agent.eval()

    obs, info = env.reset(seed=args.trace_seed)

    obs_list = [obs.copy()]
    feat_list = []
    logits_list = []
    probs_list = []
    action_list = []
    reward_list = []
    distance_list = [float(info.get("distance", np.nan))]
    cur_idx_list = [int(info.get("cur_idx", -1))]
    goal_idx = int(info.get("goal_idx", -1))

    done = False
    truncated = False
    ep_return = 0.0

    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            feat = agent.get_features(obs_tensor)
            logits = agent.actor(feat)
            probs = torch.softmax(logits, dim=-1)

            if args.trace_action_mode == "greedy":
                action = torch.argmax(logits, dim=1).item()
            elif args.trace_action_mode == "sample":
                action = Categorical(logits=logits).sample().item()
            else:
                raise ValueError(f"Unknown trace_action_mode: {args.trace_action_mode}")

        next_obs, reward, done, truncated, info = env.step(action)

        feat_list.append(feat.squeeze(0).cpu().numpy())
        logits_list.append(logits.squeeze(0).cpu().numpy())
        probs_list.append(probs.squeeze(0).cpu().numpy())
        action_list.append(action)
        reward_list.append(float(reward))
        distance_list.append(float(info.get("distance", np.nan)))
        cur_idx_list.append(int(info.get("cur_idx", -1)))
        obs_list.append(next_obs.copy())

        ep_return += float(reward)
        obs = next_obs

    env.close()

    if was_training:
        agent.train()

    obs_arr = np.stack(obs_list[:-1], axis=0) if len(obs_list) > 1 else np.empty((0, 16, 1024), dtype=np.float32)
    feats_arr = np.stack(feat_list, axis=0) if len(feat_list) > 0 else np.empty((0, 64), dtype=np.float32)
    logits_arr = np.stack(logits_list, axis=0) if len(logits_list) > 0 else np.empty((0, 4), dtype=np.float32)
    probs_arr = np.stack(probs_list, axis=0) if len(probs_list) > 0 else np.empty((0, 4), dtype=np.float32)
    actions_arr = np.array(action_list, dtype=np.int64)
    rewards_arr = np.array(reward_list, dtype=np.float32)
    dists_arr = np.array(distance_list, dtype=np.float32)
    cur_idx_arr = np.array(cur_idx_list, dtype=np.int64)

    mode_name = (
        "blind" if not args.use_observation else
        "goalonly" if args.goal_only else
        "full"
    )
    stem = f"iter{iteration:04d}__{mode_name}__seed{args.trace_seed}__{args.trace_action_mode}"

    np.savez(
        os.path.join(trace_dir, f"{stem}_trace.npz"),
        obs=obs_arr,
        features=feats_arr,
        logits=logits_arr,
        action_probs=probs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        distances=dists_arr,
        cur_idx=cur_idx_arr,
        goal_idx=np.array(goal_idx, dtype=np.int64),
        episode_return=np.array(ep_return, dtype=np.float32),
        success=np.array(float(info.get("success", 0.0)), dtype=np.float32),
        final_distance=np.array(float(info.get("final_distance", np.nan)), dtype=np.float32),
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.1, 1.2, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])

    env_plot = gym.make(
        args.env_id,
        use_observation=args.use_observation,
        goal_only=args.goal_only,
    )
    draw_grid_trajectory(ax1, env_plot, cur_idx_arr, goal_idx)
    env_plot.close()

    plot_action_probs(ax2, probs_arr)
    plot_step_table(ax3, actions_arr, rewards_arr, dists_arr[1:], cur_idx_arr)
    plot_obs_heatmap(ax4, obs_arr)
    plot_feature_heatmap(ax5, feats_arr)

    fig.suptitle(
        f"Training trace | iter={iteration} | mode={mode_name} | action_mode={args.trace_action_mode} | "
        f"return={ep_return:.3f} | success={float(info.get('success', 0.0)):.0f} | "
        f"final_dist={float(info.get('final_distance', np.nan)):.3f}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(trace_dir, f"{stem}_summary.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"[TRACE] Saved figure to: {fig_path}")   
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    success_window = deque(maxlen=100)
    return_window = deque(maxlen=100)
    distance_window = deque(maxlen=100)
    length_window = deque(maxlen=100)
    all_train_returns = []
    all_train_lengths = []
    all_train_successes = []
    all_train_final_distances = []
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, seed=args.seed + i,
                use_observation=args.use_observation,
                goal_only=args.goal_only,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None:
                        continue

                    ep_return = None
                    ep_length = None
                    final_distance = None
                    success = None

                    if "episode" in info:
                        ep_return = np.asarray(info["episode"]["r"]).item()
                        ep_length = np.asarray(info["episode"]["l"]).item()

                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)

                        return_window.append(ep_return)
                        length_window.append(ep_length)

                        writer.add_scalar("charts/episodic_return_mean_100", np.mean(return_window), global_step)
                        writer.add_scalar("charts/episodic_length_mean_100", np.mean(length_window), global_step)

                    if "final_distance" in info:
                        final_distance = float(info["final_distance"])
                        writer.add_scalar("charts/final_distance", final_distance, global_step)

                        distance_window.append(final_distance)
                        writer.add_scalar("charts/final_distance_mean_100", np.mean(distance_window), global_step)
                        writer.add_scalar("charts/final_distance_median_100", np.median(distance_window), global_step)

                    if "success" in info:
                        success = float(info["success"])
                        writer.add_scalar("charts/success", success, global_step)

                        success_window.append(success)
                        writer.add_scalar("charts/success_rate_100", np.mean(success_window), global_step)

                    # store every finished episode for final summary
                    if ep_return is not None:
                        all_train_returns.append(ep_return)
                    if ep_length is not None:
                        all_train_lengths.append(ep_length)
                    if final_distance is not None:
                        all_train_final_distances.append(final_distance)
                    if success is not None:
                        all_train_successes.append(success)

                    if ep_return is not None and final_distance is not None and success is not None:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={ep_return:.3f}, "
                            f"success={success:.0f}, "
                            f"final_distance={final_distance:.3f}"
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.trace_every > 0 and (iteration == 1 or iteration % args.trace_every == 0 or iteration == args.num_iterations):
            save_trace_rollout(agent, args, run_name, iteration, device)
        
    save_path = f"models/{run_name}_ppo_agent.pt"
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "args": vars(args)}, save_path)
    print(f"Model saved to {save_path}")
    
    print("\n" + "=" * 50)
    print("TRAINING EPISODE SUMMARY")
    print("=" * 50)
    print(f"Finished episodes   : {len(all_train_returns)}")

    if len(all_train_returns) > 0:
        print(f"Success rate        : {np.mean(all_train_successes):.4f}")
        print(f"Mean return         : {np.mean(all_train_returns):.4f}")
        print(f"Mean ep length      : {np.mean(all_train_lengths):.4f}")
        print(f"Mean final distance : {np.mean(all_train_final_distances):.4f}")
        print(f"Median final dist   : {np.median(all_train_final_distances):.4f}")

    last_n = min(1000, len(all_train_returns))
    if last_n > 0:
        print("\nLAST 1000 TRAINING EPISODES SUMMARY")
        print("=" * 50)
        print(f"Episodes used       : {last_n}")
        print(f"Success rate        : {np.mean(all_train_successes[-last_n:]):.4f}")
        print(f"Mean return         : {np.mean(all_train_returns[-last_n:]):.4f}")
        print(f"Mean ep length      : {np.mean(all_train_lengths[-last_n:]):.4f}")
        print(f"Mean final distance : {np.mean(all_train_final_distances[-last_n:]):.4f}")
        print(f"Median final dist   : {np.median(all_train_final_distances[-last_n:]):.4f}")
        
    os.makedirs("train_results", exist_ok=True)
    np.savez(
        f"train_results/{run_name}_train_metrics.npz",
        returns=np.array(all_train_returns, dtype=np.float32),
        lengths=np.array(all_train_lengths, dtype=np.float32),
        successes=np.array(all_train_successes, dtype=np.float32),
        final_distances=np.array(all_train_final_distances, dtype=np.float32),
    )
    print(f"Training metrics saved to train_results/{run_name}_train_metrics.npz")

    envs.close()
    writer.close()