# PortSentinel: Reinforcement Learning for Network Port Security

## Project Overview

**PortSentinel-PPO** is a self-contained Jupyter notebook that implements a Gym-style environment and a PPO-based Reinforcement Learning agent for network port security. Within the notebook, you’ll find:

* **Custom Environment**: `AdvancedNetworkEnv` simulates port activity, packet history, and threat stages.
* **PPO Agent Setup**: Full Stable-Baselines3 PPO configuration, including callbacks for live monitoring.
* **Training & Evaluation**: End-to-end code for training, visualization of learning curves, and comprehensive evaluation.

Everything—all code, experiments, visualizations, and results—is contained in a single `PortSentinel.ipynb` file for easy exploration and reproduction.

## Repository Structure

```
├── PortSentinel.ipynb      # Main notebook with code, experiments, and charts
├── RL_Formulation.pdf      # Mathematical formulation of the environment and rewards
├── trained_model.zip       # Saved PPO model checkpoint (after training)
├── logs/                   # TensorBoard and callback logs (auto-generated)
├── requirements.txt        # Python dependencies for running the notebook
└── README.md               # This documentation file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aditya-Codes-247/PortSentinel.git
   cd PortSentinel
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

*Alternatively*, run this cell at the top of the notebook:

```python
!pip install stable-baselines3 gym matplotlib pyvirtualdisplay tqdm
```

## Training & Hyperparameters

Hyperparameters for the PPO agent align with the notebook’s model definition:

* **n\_steps**: 2048
* **Batch size**: 128
* **Epochs per update (n\_epochs)**: 10
* **Learning rate**: 2.5e-4
* **Clip range (epsilon)**: 0.2
* **Discount factor (gamma)**: 0.99
* **GAE parameter (lambda)**: 0.95
* **Entropy coefficient (ent\_coef)**: 0.02
* **Value loss coefficient (vf\_coef)**: 0.7
* **Max gradient norm**: 0.5
* **Seed**: `RANDOM_SEED`
* **Device**: e.g., `cpu` or `cuda`
* **TensorBoard log dir**: `LOG_DIR`

Example in-notebook instantiation:

This project models network port security as an MDP with the following specification (see RL\_Formulation.pdf) citeturn1file0:

* **Common Ports:** 80, 443, 8080, 22, 53
* **Suspicious Ports:** 4444, 31337, 6667
* **State Space (16 dimensions):**

  * Suspicion level (0–100)
  * Maximum suspicion cap
  * Current port (normalized)
  * Packet mean size (normalized)
  * Packet max size (normalized)
  * Fraction of large packets
  * Unique port count (normalized)
  * Time normalization (progress in episode)
  * One-hot encoding of last action (5 possible actions)
  * Last four port values (normalized)
* **Action Space (5 discrete actions):**
  0\. Send small packet (200 bytes)

  1. Send large packet (1500 bytes)
  2. Delay (no packet)
  3. Change port (random common port)
  4. Stealth combo (delay → small packet → change port)
* **Dynamics and Detection:**

  * Packet size check: trigger if enough recent packets exceed a size threshold.
  * Port scan check: trigger if too many unique recent ports.
  * Suspicious port check: probabilistic detection when on a suspicious port.
  * Episodes terminate on detection or at a maximum step count.
* **Reward Function:**

  * **Detection penalty:** -100 minus 0.5 times remaining steps.
  * **Survival reward:** base +0.2, +0.5 if on a common port, -0.5 if action=change port, +0.2 for diverse history.
  * **Completion bonus:** +10 if the agent survives to the max step count.

## Training & Hyperparameters

We train using PPO with parameters from the formulation citeturn1file0:

* Discount factor (gamma): 0.99
* GAE parameter (lambda): 0.95
* Clipping range (epsilon): 0.2
* Value loss coefficient (c1): 0.7
* Entropy coefficient (c2): 0.02
* Learning rate: 3e-4
* Batch size: 64
* Epochs per update: 10
* Total timesteps: configurable (e.g., 1e6)

Example in-notebook instantiation:

```python
model = PPO(
    "MlpPolicy", env,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    vf_coef=0.7,
    ent_coef=0.02,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    verbose=1,
)
model.learn(total_timesteps=1_000_000, callback=TrainingCallback())
model.save("trained_model.zip")
```

## Results & Visuals

The notebook includes detailed visualizations from both training and evaluation phases:

**Training Phase Plots**

* **Training Reward Progression**: Episode rewards (blue) with a moving average (red) to show learning trends.
* **Suspicion Level Distribution**: Histogram of final suspicion levels across episodes.
* **Action Selection Distribution**: Bar chart of action frequencies (Small Packet, Large Packet, Delay, Change Port, Stealth Combo).
* **Detection Causes Breakdown**: Pie chart showing proportions of `Port Scan`, `Packet Size`, and `Survived` outcomes.
* **Top Port Usage Frequency**: Bar chart of the five most-used ports (e.g., 8080, 53, 22, 443, 80).
* **Reward vs. Suspicion Correlation**: Scatter plot mapping episode reward against final suspicion level.
* **Success Rate Over Time**: Rolling-window line plot of agent success rate, with a 50% benchmark.
* **Reward Distribution by Detection Cause**: Box plots of rewards grouped by detection cause categories.
* **Action Diversity Over Time**: Rolling-window plot showing the number of unique actions used per window.
* **Common Port Strategy Evolution**: Time series of the ratio of common-port usage across episodes.

**Live Performance Dashboard (Training)**

* Real-time plots of reward and suspicion over steps.
* Live action distribution and port usage frequency panels.
* Agent status overlay indicating detection or successful evasion.

**Evaluation Phase Plots**

* **Evaluation Reward Distribution**: Histogram of total episode rewards with mean overlay.
* **Detection Rate**: Pie chart showing percentages of `Evaded Detection` vs. `Detected` episodes.
* **Detection Causes Breakdown (Eval)**: Pie chart of `Survived` episodes in evaluation.
* **Action Selection Distribution (Eval)**: Bar chart of action counts during evaluation episodes.
* **Episode Length Distribution**: Histogram of episode lengths with mean marker.
* **Unique Ports Used Distribution**: Histogram of the number of distinct ports visited per episode with mean overlay.

All figures are rendered inline in the notebook, allowing easy reference and interpretation. Images are automatically saved under the `logs/` directory for archival.

## Contributing

Contributions via GitHub issues or pull requests are welcome—add new threat scenarios, tweak hyperparameters, or enhance visualizations directly in the notebook.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## References

* Schulman et al., 2017. *Proximal Policy Optimization Algorithms*.
* Stable-Baselines3 PPO documentation: [https://stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
