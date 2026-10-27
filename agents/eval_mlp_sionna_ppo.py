import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.register_sionna_env


@dataclass
class Args:
    model_path: str
    env_id: str = "SionnaGrid2D-v0"
    eval_episodes: int = 1000
    seed: int = 42
    cuda: bool = True

    # must match training env mode
    use_observation: bool = True
    goal_only: bool = False
    obs_mode: str = "raw"


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()

        obs_shape = obs_space.shape
        n_actions = act_space.n

        self.is_sequence = (len(obs_shape) == 2)

        if self.is_sequence:
            c_in, seq_len = obs_shape
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
            feat_dim = 64

        elif len(obs_shape) == 1:
            in_dim = obs_shape[0]
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            feat_dim = 64

        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")

        self.actor = layer_init(nn.Linear(feat_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(feat_dim, 1), std=1.0)

    def get_features(self, x):
        x = x.float()
        return self.encoder(x)

    def get_value(self, x):
        z = self.get_features(x)
        return self.critic(z)

    def get_logits(self, x):
        z = self.get_features(x)
        return self.actor(z)

    def get_action_and_value(self, x, action=None):
        z = self.get_features(x)
        logits = self.actor(z)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(z)


def evaluate(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    env = gym.make(
        args.env_id,
        use_observation=args.use_observation,
        goal_only=args.goal_only,
        obs_mode=args.obs_mode,
    )

    print("Env observation shape:", env.observation_space.shape)
    print("Env action space:", env.action_space)

    agent = Agent(env.observation_space, env.action_space).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["state_dict"])
    else:
        agent.load_state_dict(checkpoint)

    agent.eval()

    returns = []
    lengths = []
    successes = []
    final_distances = []

    for ep in range(args.eval_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (done or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits = agent.get_logits(obs_tensor)
                action = torch.argmax(logits, dim=1).item()   # greedy eval

            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(float(info.get("success", 0.0)))
        final_distances.append(float(info.get("final_distance", np.nan)))

        print(
            f"Ep {ep+1:4d} | "
            f"Return {ep_return:7.3f} | "
            f"Len {ep_len:3d} | "
            f"Success {float(info.get('success', 0.0)):.0f} | "
            f"FinalDist {float(info.get('final_distance', np.nan)):.3f}"
        )

    env.close()

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes           : {args.eval_episodes}")
    print(f"Success rate       : {np.mean(successes):.4f}")
    print(f"Mean final distance: {np.mean(final_distances):.4f}")
    print(f"Median final dist  : {np.median(final_distances):.4f}")
    print(f"Mean return        : {np.mean(returns):.4f}")
    print(f"Mean episode length: {np.mean(lengths):.4f}")

    save_dir = os.path.join("eval_results", os.path.splitext(os.path.basename(args.model_path))[0])
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "eval_metrics.npz"),
        returns=np.array(returns, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.float32),
        successes=np.array(successes, dtype=np.float32),
        final_distances=np.array(final_distances, dtype=np.float32),
    )
    print(f"Saved results to: {save_dir}")


if __name__ == "__main__":
    evaluate(tyro.cli(Args))