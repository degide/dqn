"""
Deep Q-Network (DQN) Evaluation & Gameplay Script
Environment: SpaceInvadersNoFrameskip-v4 (Atari)
Framework: Stable Baselines3 + Gymnasium + ALE

Usage:
    python play.py --model dqn_model.zip --episodes 5 --render
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Disable GPU for evaluation. Comment out to enable GPU if available.

import argparse
import time
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


# CONFIGURATION

ENV_ID          = "ALE/SpaceInvaders-v5"
N_STACK         = 4       # Must match the value used during training
DEFAULT_MODEL   = "best_model/best_model.zip"   # Path to the saved model
N_EPISODES      = 5       # Number of evaluation episodes
RENDER_DELAY    = 0.03    # Seconds between rendered frames (controls visual speed)


# GREEDY Q-POLICY EVALUATION
#   In Stable Baselines3, passing `deterministic=True` to
#   model.predict() implements a Greedy Q-Policy:
#     a* = argmax_a Q(s, a; θ)
#   This eliminates all exploration (ε = 0) and always selects
#   the action with the highest estimated Q-value.

def evaluate_agent(
    model_path: str,
    n_episodes: int,
    render: bool,
) -> None:
    """
    Load a trained DQN model and evaluate it over a specified
    number of episodes using a Greedy Q-Policy (deterministic=True).

    Parameters
    ----------
    model_path : str
        Path to the saved .zip model file.
    n_episodes : int
        Number of complete episodes to run.
    render : bool
        Whether to visualise the game using env.render().
    """

    # 1. Load the trained model
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}\n")

    model = DQN.load(model_path)

    # 2. Set up the Atari environment
    #    render_mode="human" opens the GUI window via ALE/Pygame
    render_mode = "human" if render else "rgb_array"

    env = make_atari_env(ENV_ID, n_envs=1, seed=0,
                         env_kwargs={"render_mode": render_mode})
    env = VecFrameStack(env, n_stack=N_STACK)

    # 3. Evaluation loop
    episode_rewards  = []
    episode_lengths  = []

    print(f"Running {n_episodes} evaluation episode(s) with Greedy Q-Policy …\n")

    for ep in range(1, n_episodes + 1):
        obs          = env.reset()
        done         = False
        total_reward = 0.0
        step_count   = 0

        while not done:
            # Greedy Q-Policy: deterministic=True
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, terminated, info = env.step(action)

            total_reward += float(reward.sum())
            step_count   += 1
            done          = bool(terminated.any())

            if render:
                # env.render() is called implicitly when render_mode="human"
                time.sleep(RENDER_DELAY)

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        print(
            f"  Episode {ep:>2}/{n_episodes}  |  "
            f"Reward: {total_reward:8.2f}  |  "
            f"Steps: {step_count}"
        )

    env.close()

    # 4. Summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Environment       : {ENV_ID}")
    print(f"  Model             : {model_path}")
    print(f"  Policy mode       : Greedy Q-Policy (deterministic=True)")
    print(f"  Episodes          : {n_episodes}")
    print(f"  Mean Reward       : {np.mean(episode_rewards):.2f}")
    print(f"  Std  Reward       : {np.std(episode_rewards):.2f}")
    print(f"  Max  Reward       : {np.max(episode_rewards):.2f}")
    print(f"  Min  Reward       : {np.min(episode_rewards):.2f}")
    print(f"  Mean Episode Len  : {np.mean(episode_lengths):.1f} steps")
    print(f"{'='*60}\n")


# COMMAND-LINE INTERFACE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on an Atari environment."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to trained model .zip (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EPISODES})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the game GUI during evaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
    )