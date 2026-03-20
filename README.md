# Formative 3 — Deep Q-Network (DQN) on Atari: SpaceInvaders

## Environment
**ALE/SpaceInvaders-v5** — The agent controls a spaceship and must eliminate waves of alien invaders while avoiding their projectiles. The reward signal is clear and dense, making it well-suited for observing DQN convergence behaviour.

---

## Repository Structure

```
├── train.py               # DQN training script
├── play.py                # Evaluation / gameplay script  
├── dqn_model.zip          # Final saved model
├── best_model/
│   └── best_model.zip     # Best model (saved by EvalCallback)
├── checkpoints/           # Periodic checkpoints during training
├── training_log.csv       # Per-episode reward & length log
├── tensorboard_logs/      # TensorBoard training curves
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
AutoROM --accept-license
```

---

## Usage

### Training
```bash
python train.py
```

### Evaluation (with GUI rendering)
```bash
python play.py --model best_model/best_model.zip --episodes 5 --render
```

---

## Policy Comparison: CnnPolicy vs MlpPolicy

| Policy     | Architecture               | Suited for Atari? | Notes                                                  |
|------------|----------------------------|-------------------|--------------------------------------------------------|
| CnnPolicy  | Convolutional Neural Net   | Yes             | Processes raw 84×84 pixel frames; extracts spatial features |
| MlpPolicy  | Multilayer Perceptron      | No              | Requires flat feature vectors; loses spatial structure |

**Conclusion:** CnnPolicy is strongly preferred for pixel-based Atari environments. MlpPolicy applied to stacked frames treats each pixel as an independent feature, which is computationally intractable and semantically inappropriate.

---

## Hyperparameter Tuning — Experiments

> Each group member must document **10 experiments** with different hyperparameter combinations.
> The table below provides the schema and example experiments. Replace with your actual observed values.

### Member: [YOUR NAME]

| # | `lr` | `gamma` | `batch_size` | `ε_start` | `ε_end` | `ε_decay (fraction)` | Noted Behaviour |
|---|------|---------|--------------|-----------|---------|----------------------|-----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Baseline config. Stable learning; reward increases gradually after ~200k steps. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Higher LR causes unstable Q-value estimates; reward oscillates. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | LR too low; learning is extremely slow; minimal reward improvement by 1M steps. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.10 | Lower γ makes agent more myopic; prioritises immediate rewards, shorter survival. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 0.10 | Higher γ improves long-term planning; agent learns to dodge more strategically. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | Larger batch reduces variance in gradient updates; smoother reward curve. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.10 | Smaller batch introduces noisy gradients; instability observed early in training. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | Higher ε_end retains more exploration; agent explores longer but converges slower. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.20 | Slower ε decay; prolonged exploration phase yields better policy at convergence. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.05 | Rapid ε decay; agent exploits early but with an underdeveloped policy; reward plateaus. |

> **Best Configuration:** Experiment 5 or 9 — high γ combined with slow ε decay produced the highest mean episode reward, as the agent balanced exploration with long-term value optimisation.

---

## Gameplay Demo

> [Insert link to your screen recording of `play.py` output here]

---

## Key Insights

- **Learning Rate** is the most sensitive hyperparameter: too high causes divergence, too low causes impractical training times.
- **Gamma (γ)** controls temporal horizon. For SpaceInvaders, where surviving longer correlates with higher score, high γ (≥ 0.99) is beneficial.
- **Epsilon decay** must be balanced — premature exploitation with a weak policy leads to early convergence on a sub-optimal local policy.
- **Batch size** affects gradient variance: larger batches stabilise training but may slow wall-clock time per update.