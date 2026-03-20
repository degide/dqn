"""
Deep Q-Network (DQN) Training Script
Environment: ALE/SpaceInvaders-v5 (Atari)
Framework: Stable Baselines3 + Gymnasium + ALE
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Disable GPU for evaluation. Comment out to enable GPU if available.

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
import numpy as np

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

# 0. CONFIGURATION
SAVE_DIR = "./"

# 1.  CUSTOM LOGGING CALLBACK
#     Records mean episode reward and episode length per rollout
class TrainingLogger(BaseCallback):
    """
    A lightweight callback that logs reward trends and episode
    lengths to the console and appends them to a CSV file for
    post-hoc analysis and hyperparameter-table documentation.
    """

    def __init__(self, log_path: str = "training_log.csv", verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        # Initialise CSV
        with open(self.log_path, "w") as f:
            f.write("timestep,mean_reward,mean_ep_length\n")

    def _on_step(self) -> bool:
        # SB3 stores per-episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)

                if self.verbose:
                    print(
                        f"[Step {self.num_timesteps:>8}]  "
                        f"Episode Reward: {ep_rew:7.2f}  |  "
                        f"Episode Length: {ep_len}"
                    )

                with open(self.log_path, "a") as f:
                    f.write(
                        f"{self.num_timesteps},"
                        f"{ep_rew:.4f},"
                        f"{ep_len}\n"
                    )
        return True  # returning False would abort training



# 2.  ENVIRONMENT FACTORY
#     Applies standard Atari pre-processing:
#       • Grayscale + resize to 84×84
#       • Frame-skip = 4  (NoFrameskip variant handles this)
#       • Frame-stack = 4  (temporal context for the CNN)

ENV_ID = "ALE/SpaceInvaders-v5"
N_ENVS = 4          # Parallel environments for faster data collection
N_STACK = 4         # Stacked frames fed to the CNN


def make_env(n_envs: int = N_ENVS, seed: int = 42) -> VecFrameStack:
    """
    Build a vectorised, pre-processed Atari environment.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    VecFrameStack
        A frame-stacked vectorised environment.
    """
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env



# 3.  HYPERPARAMETER CONFIGURATION
#     Modify these values to run each of the 10 experiments.
#     Document observations in the hyperparameter table.

HYPERPARAMS = {
    # Core learning parameters
    "learning_rate":        1e-4,   # α  — step size for gradient descent
    "gamma":                0.99,   # γ  — discount factor for future rewards
    "batch_size":           32,     # mini-batch size sampled from replay buffer

    # Exploration (ε-greedy)──
    "exploration_fraction": 0.10,   # fraction of training over which ε decays
    "exploration_initial_eps": 1.0, # ε_start — fully random at the beginning
    "exploration_final_eps":   0.01,# ε_end   — minimum exploration rate

    # Replay buffer & target network
    "buffer_size":          100_000,# experience replay buffer capacity
    "learning_starts":      10_000, # steps before learning begins
    "target_update_interval": 1_000,# steps between target-network hard updates
    "train_freq":           4,      # environment steps between each gradient update

    # Policy architecture
    #   "CnnPolicy": Processes raw pixels; suited for Atari (recommended)
    #   "MlpPolicy": Processes flat feature vectors; less suitable here
    "policy": "CnnPolicy",
}

TOTAL_TIMESTEPS = 30_000   # Increase for better convergence (e.g., 10M)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "dqn_model")  # Saved as dqn_model.zip by SB3


# 4.  POLICY COMPARISON HELPER
#     Runs a short trial for both CnnPolicy and MlpPolicy and
#     reports which achieves a higher mean reward, fulfilling
#     the assignment requirement to compare both architectures.

def compare_policies(timesteps: int = 50_000) -> None:
    """
    Train both CnnPolicy and MlpPolicy briefly and report
    mean episode reward for a qualitative comparison.
    """
    print("\n" + "=" * 60)
    print("POLICY COMPARISON: CnnPolicy vs MlpPolicy")
    print("=" * 60)

    results = {}
    for policy in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n>  Training with {policy} for {timesteps} steps …")
        env = make_env(n_envs=1)

        # MlpPolicy requires flattened observations — wrap accordingly
        model = DQN(
            policy=policy,
            env=env,
            learning_rate=HYPERPARAMS["learning_rate"],
            gamma=HYPERPARAMS["gamma"],
            batch_size=HYPERPARAMS["batch_size"],
            buffer_size=10_000,          # smaller buffer for quick trial
            learning_starts=1_000,
            device="cpu",               # Quadro P1000 (sm_61) incompatible with PyTorch 2.x
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)

        # Evaluate over 5 episodes
        obs = env.reset()
        ep_rewards = []
        current_reward = 0.0
        for _ in range(5 * 500):         # rough upper bound
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            current_reward += reward.sum()
            if done.any():
                ep_rewards.append(current_reward)
                current_reward = 0.0
                if len(ep_rewards) >= 5:
                    break
        env.close()

        mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        results[policy] = mean_rew
        print(f"   {policy} — Mean Reward (5 episodes): {mean_rew:.2f}")

    winner = max(results, key=results.get)
    print(f"\n[OK]  Recommended policy: {winner} "
          f"(mean reward = {results[winner]:.2f})\n")


# 5.  MAIN TRAINING ROUTINE

def train() -> None:
    """
    Instantiate and train the DQN agent, then save the model.
    """
    print("\n" + "=" * 60)
    print(f"TRAINING DQN AGENT  —  {ENV_ID}")
    print("=" * 60)
    print("Hyperparameters:")
    for k, v in HYPERPARAMS.items():
        print(f"  {k:<30} = {v}")
    print("=" * 60 + "\n")

    # 5a. Build training and evaluation environments
    train_env = make_env(n_envs=N_ENVS, seed=42)
    eval_env  = make_env(n_envs=1,      seed=99)

    # 5b. Define callbacks
    logger_cb = TrainingLogger(log_path=os.path.join(SAVE_DIR, "training_log.csv"), verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best_model"),
        log_path=os.path.join(SAVE_DIR, "eval_logs"),
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        name_prefix="dqn_checkpoint",
        verbose=1,
    )

    # 5c. Instantiate the DQN model
    model = DQN(
        policy=HYPERPARAMS["policy"],
        env=train_env,
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        batch_size=HYPERPARAMS["batch_size"],
        exploration_fraction=HYPERPARAMS["exploration_fraction"],
        exploration_initial_eps=HYPERPARAMS["exploration_initial_eps"],
        exploration_final_eps=HYPERPARAMS["exploration_final_eps"],
        buffer_size=HYPERPARAMS["buffer_size"],
        learning_starts=HYPERPARAMS["learning_starts"],
        target_update_interval=HYPERPARAMS["target_update_interval"],
        train_freq=HYPERPARAMS["train_freq"],
        optimize_memory_usage=False,
        device="cpu",                    # Change if GPU is available
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )

    # 5d. Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[logger_cb, eval_cb, checkpoint_cb],
        log_interval=10,
    )

    # 5e. Save the final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n[OK]  Model saved -> {MODEL_SAVE_PATH}.zip")
    print(f"[OK]  Training log -> {os.path.join(SAVE_DIR, 'training_log.csv')}")
    print(f"[OK]  Best model   -> {os.path.join(SAVE_DIR, 'best_model/best_model.zip')}\n")

    train_env.close()
    eval_env.close()



if __name__ == "__main__":
    # Uncomment the line below to run a brief policy comparison first
    # compare_policies(timesteps=50_000)

    train()