
# Deep Q-Network (DQN) Implementation with Advanced Features

A comprehensive PyTorch implementation of Deep Q-Network and its variants including Double DQN, Prioritized Experience Replay, and Multi-step Learning for reinforcement learning education.


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm Details](#algorithm-details)
    - [Base DQN](#base-dqn)
    - [Experience Replay](#experience-replay)
    - [Target Network](#target-network)
    - [Prioritized Experience Replay (PER)](#prioritized-experience-replay-per)
    - [Multi-step Learning](#multi-step-learning)
    - [Double DQN (DDQN)](#double-dqn-ddqn)
    - [Dueling DQN](#dueling-dqn)
- [Usage Examples](#usage-examples)
- [References](#references)



## Features

- **Base DQN**: Vanilla Deep Q-Network with experience replay and target networks
- **Double DQN**: Reduces overestimation bias in Q-value updates
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **Multi-step Learning**: Uses n-step returns for faster learning
- **Dueling Architecture**: Separates state value and advantage estimation
- **Flexible Replay Buffers**: Uniform, prioritized(PER), or no replay options
- **Target Network Updates**: Both soft (Polyak) and hard update strategies





## Algorithm Details

### Base DQN

The foundation of our implementation follows the classic DQN formulation:

**Bellman Update:**
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_{t+1} + \gamma \max_{a} Q(s_{t+1},a) - Q(s_t,a_t)]$$

**Loss Function:**
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a))^2\right]$$

Where $\theta^-$ represents the target network parameters and $\mathcal{D}$ is the experience replay buffer.


However, the vanilla DQN often diverges due to the following reasons:
1. temporal correlations between samples 
2. non-stationary targets

These problems are solved by introducing the following tricks:
1. Experience Replay
2. The use of target network

Here, experience replay helps to overcome the temporal correlation problem, and the target network helps to overcome the non-stationary target problem.

### Experience Replay
Experience replay addresses two critical problems in neural Q-learning:

1. **Strongly temporally correlated updates**: Sequential observations are highly correlated
2. **Catastrophic Forgetting**: Rare but important experiences are quickly overwritten

Benefits of using Experience Replay:
- Breaks temporal correlations between consecutive samples
- Enables efficient mini-batch learning
- Improves sample efficiency through experience reuse

### Target network
If the target function is changing frequently, it becomes more difficult for it to be trained.

In this idea, the previous parameter $\hat{\theta}$ is fixed to **Target $\hat{Q}$ network** and updates parameters $\theta$ only on **Behavior $Q$-network**

**Hard Update** (every N steps):
$\theta^- \leftarrow \theta$

**Soft Update** (Polyak averaging):
$\theta^- \leftarrow (1-\tau)\theta^- + \tau\theta$ 
where $\tau \in (0,1]$ is the update rate.


### Prioritized Experience Replay (PER)

PER samples transitions based on their temporal-difference (TD) error magnitude:

**Priority Calculation:**
$p_i = (|\delta_i| + \varepsilon)^\alpha$

**Sampling Probability:**
$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$

**Importance Sampling Weights:**
$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$

Where:

- $\alpha$ controls prioritization strength (0 = uniform, 1 = full prioritization)
- $\beta$ corrects sampling bias (annealed from $\beta_0$ to 1.0)
- $\varepsilon$ prevents zero priorities



### Multi Step Learning [3]
Forward-view multi-step targets can be used.
The truncated n-step return
from a given state $S_t$ can be defined as 
$$R_t^{(n)} ≡ \sum_{k=0}^{n-1}γ_t^{(k)}R_{t+k+1}$$

A multi-step variant of DQN can be defined by the minimizing alternative loss:
$$(R_t^{(n)}+ γ^{(n)}_t \max_{a'}q_θ(S_{t+n}, a') − q_θ(S_t, A_t))^2$$

### Double DQN (DDQN)

### Dueling DQN

## Usage Examples

### Basic Training

The code uses gymnasium's `LunarLander-v3` environment by default. You can run the training script with various configurations or you can change the environment in the code:

```bash
# Standard DQN with default settings
python dqn.py

# Custom hyperparameters example
python dqn.py --lr 5e-4 --batch_size 128 --gamma 0.995 --epsilon_decay 10000
```

### Ablation Studies examples

```bash
# Compare replay buffer strategies
python dqn.py --replay_buffer uniform --episodes 500
python dqn.py --replay_buffer prioritized --episodes 500
python dqn.py --replay_buffer none --episodes 500

# Compare target update strategies
python dqn.py --target_update_type soft --tau 0.001
python dqn.py --target_update_type hard --hard_update_freq 200

# Test n-step learning
python dqn.py --nstep 1  # Standard 1-step
python dqn.py --nstep 3  # 3-step returns
python dqn.py --nstep 5  # 5-step returns

# Turnoff the replay buffer or the target network
python dqn.py --no_replay_buffer
python dqn.py --no_target_network


```
## Configuration Options

| Parameter                                        | Description                                          | Default       | Typical Range / Choices          |
| :----------------------------------------------- | :--------------------------------------------------- | :------------ | :------------------------------- |
| `--episodes`                                     | Total training episodes.                             | `1000`        | `1` – ∞                          |
| `--batch_size`                                   | Transitions sampled per optimization step.           | `128`         | `32` – 1024                      |
| `--gamma`                                        | Discount factor γ.                                   | `0.99`        | `0.90` – `0.999`                 |
| `--epsilon_start`                                | Initial ε for ε-greedy policy.                       | `0.9`         | `0.1` – `1.0`                    |
| `--epsilon_end`                                  | Final ε after decay.                                 | `0.05`        | `0.01` – `0.2`                   |
| `--epsilon_decay`                                | Steps over which ε decays exponentially.             | `5000`        | `10³` – `10⁶`                    |
| `--lr`                                           | Adam learning rate.                                  | `1e-4`        | `1e-6` – `1e-3`                  |
| `--learning_starts`                              | Steps before first optimization.                     | `100`         | `0` – `10⁴`                      |
| `--memory_size`                                  | Replay buffer capacity.                              | `10000`       | `10³` – `10⁶+`                   |
| `--replay_buffer`                                | Buffer type.                                         | `prioritized` | `none`, `uniform`, `prioritized` |
| `--per-alpha`                                    | Prioritization exponent α.                           | `0.6`         | `0.0` – `1.0`                    |
| `--per-beta0`                                    | Initial importance-sampling exponent β.              | `0.5`         | `0.0` – `1.0`                    |
| `--per-beta1`                                    | Final β.                                             | `1.0`         | `0.0` – `1.0`                    |
| `--per-beta-steps`                               | Annealing steps from β₀ to β₁.                       | `50000`       | `0` – `10⁶`                      |
| `--per-eps`                                      | Priority additive ε to avoid zero prob.              | `1e-5`        | `1e-8` – `1e-2`                  |
| `--target-network / --no-target-network`         | Enable separate target network.                      | `True`        | Boolean                          |
| `--target_update_type`                           | Target update rule.                                  | `soft`        | `soft`, `hard`                   |
| `--tau`                                          | Soft-update coefficient τ (Polyak).                  | `0.005`       | `0.001` – `0.1`                  |
| `--hard_update_freq`                             | Hard-update period (steps).                          | `200`         | `10` – `10⁴`                     |
| `--nstep`                                        | Horizon k for n-step returns.                        | `1`           | `1` – `10`                       |
| `--bootstrap-on-trunc / --no-bootstrap-on-trunc` | Treat truncations as non-terminal for bootstrapping. | `True`        | Boolean                          |
| `--seed`                                         | RNG seed for reproducibility.                        | `42`          | any 32-bit int                   |


## References
DQN \
[1] https://arxiv.org/pdf/1312.5602 

PER (Prioritized Experienced Replay) \
[2] https://arxiv.org/abs/1511.05952 

Multi Step Learning \
[3-1] https://arxiv.org/pdf/1710.02298 \
[3-2] https://arxiv.org/pdf/1703.01327
