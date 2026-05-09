import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainConfig:
    total_timesteps: int = 200_000
    seed: int = 0
    traffic_density: float = 0.2
    map_name: str = "SOC"
    save_path: str = "models/ppo_metadrive"


class SmoothRewardWrapper:
    """Reward shaping wrapper focused on stability and safety.

    Adds penalties for steering/throttle jerk and strong safety penalties for
    out_of_road / crash states. This helps PPO learn smoother control near
    curves and roundabouts.
    """

    def __init__(self, env):
        self.env = env
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        self.prev_action[:] = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        steer = float(action[0])
        throttle = float(action[1])

        d_steer = abs(steer - float(self.prev_action[0]))
        d_throttle = abs(throttle - float(self.prev_action[1]))

        # Smoothness shaping.
        reward -= 0.8 * d_steer
        reward -= 0.3 * d_throttle

        # Strong safety shaping.
        if info.get("out_of_road", False):
            reward -= 25
        if info.get("crash", False):
            reward -= 50

        # Small bonus for maintaining motion.
        if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 4:
            speed = float(obs[3])
            reward += 0.15 * np.clip(speed, 0.0, 1.2)

        self.prev_action[0] = steer
        self.prev_action[1] = throttle
        return obs, reward, terminated, truncated, info

    def __getattr__(self, item):
        return getattr(self.env, item)


def make_env(cfg: TrainConfig):
    from metadrive import MetaDriveEnv

    env = MetaDriveEnv(
        {
            "use_render": False,
            "manual_control": False,
            "traffic_density": cfg.traffic_density,
            "num_scenarios": 50,
            "start_seed": cfg.seed,
            "map": cfg.map_name,
            "accident_prob": 0.0,
            "decision_repeat": 1,
        }
    )
    return SmoothRewardWrapper(env)


def train(cfg: TrainConfig):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "Missing RL dependencies. Install with: pip install stable-baselines3 torch"
        ) from exc

    env = Monitor(make_env(cfg))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        tensorboard_log="runs/ppo_metadrive",
        seed=cfg.seed,
    )

    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.save_path)
    env.close()

    print(f"Model saved to: {cfg.save_path}")


def evaluate(model_path: str, episodes: int = 3, seed: int = 100):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Missing RL dependencies. Install with: pip install stable-baselines3 torch"
        ) from exc

    cfg = TrainConfig(seed=seed)
    env = make_env(cfg)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
        print(
            f"episode={ep} reward={ep_reward:.3f} "
            f"crash={info.get('crash', False)} out_of_road={info.get('out_of_road', False)}"
        )

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for MetaDrive challenge")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traffic", type=float, default=0.2)
    parser.add_argument("--map", type=str, default="SOC")
    parser.add_argument("--save", type=str, default="models/ppo_metadrive")
    parser.add_argument("--eval", type=str, default="", help="Path to a saved model for evaluation")
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval:
        evaluate(args.eval, episodes=args.episodes, seed=args.seed)
        return

    cfg = TrainConfig(
        total_timesteps=args.timesteps,
        seed=args.seed,
        traffic_density=args.traffic,
        map_name=args.map,
        save_path=args.save,
    )
    train(cfg)


if __name__ == "__main__":
    main()
