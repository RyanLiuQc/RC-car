# Model Checkpoints and Policy Progression Log

This directory contains trained Reinforcement Learning (RL) policy model weights (`.pth` PyTorch checkpoints). This document records the progression of policy iterations, detailing reward function adjustments, empirical driving behaviors, training stability observations, and targeted improvements.

---

## 1. Model Progression and Empirical Observations

### 1.1 `a2c_policy.pth` — Baseline Iteration (First Attempt)

![A2C Baseline Policy - Wall Hugging](../docs/media/a2c_policy_baseline.gif)

* **Algorithm:** Advantage Actor-Critic (A2C, 1-Step TD)
* **Reward Formulation:** Naive baseline speed incentive combined with linear lateral displacement penalties.
* **Observed Driving Behavior:**
  * **Success:** Successfully completes laps without collisions into track boundaries.
  * **Defect:** Exhibits severe **wall-hugging behavior**. The policy settles into a sub-optimal equilibrium where it rides close to outer track boundaries rather than maintaining lane center ($d \approx 0$).

---

### 1.2 `a2c_policy_redesign_1.pth` — Reward Function Redesign (Iteration 1)

![A2C Redesign 1 - Tight Centering & Steering Jerk](../docs/media/a2c_policy_redesign_1.gif)

* **Algorithm:** Advantage Actor-Critic (A2C, 1-Step TD)
* **Reward Formulation Changes:**
  * **Removal of Arc-Length Progression:** Eliminated linear centerline delta reward ($\Delta s / \Delta t$) to prevent reward exploitation, such as policies oscillating back and forth to maximize reward accumulation.
  * **Gaussian Centering Incentive:** Replaced linear lateral penalty with a steep Gaussian decay term centered at $d=0$:
    $$\text{centering\_factor} = \exp\left(-\left(\frac{d}{\sigma}\right)^2\right)$$
* **Observed Driving Behavior:**
  * **Success:** Excellent lateral centering. The vehicle remains positioned near the track centerline ($d \approx 0$) through straightaways and curves.
  * **Defect:** High control chatter and **steering jerk**. Because an action rate / differential steering penalty ($\|a_t - a_{t-1}\|^2$) was not yet incorporated, the policy outputs rapid high-frequency steering oscillations.
* **Training Dynamics and Inconsistency:**
  * **High Variance:** Training stability is inconsistent. Repeated training runs under identical hyperparameter settings frequently suffer from performance degradation or policy collapse.
  * **Stochastic Artifact:** The performance of `a2c_policy_redesign_1.pth` was largely driven by favorable stochastic initialization and rollout variance, avoiding policy collapse during gradient descent.

---

## 2. Summary Comparison Matrix

| Model Checkpoint | Demonstration | Primary Reward Feature | Centering ($d \approx 0$) | Control Smoothness | Training Consistency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`a2c_policy.pth`** | ![A2C Baseline](../docs/media/a2c_policy_baseline.gif) | Linear speed and boundary penalty | Wall Hugging | Low Oscillations | Inconsistent |
| **`a2c_policy_redesign_1.pth`** | ![A2C Redesign 1](../docs/media/a2c_policy_redesign_1.gif) | Gaussian centering and progress coupling | Excellent ($d \approx 0$) | Severe Steering Jerk | Inconsistent / High Variance |

---

## 3. Targeted Technical Improvements

To resolve training instability, prevent policy collapse, and achieve smooth continuous vehicle control, the following technical upgrades are identified:

1. **Multi-Step Advantage Estimation (Multi-Step TD / GAE):**
   * Replace 1-step Temporal Difference error ($N=1$) with Generalized Advantage Estimation (GAE-$\lambda$) over multi-step trajectory rollouts ($T=128$). Single-step updates introduce high variance that directly contributes to policy collapse.
2. **Action Rate / Steering Jerk Penalty:**
   * Incorporate an explicit action smoothing penalty into the reward function:
     $$\mathcal{P}_{\text{jerk}} = \gamma_{\text{jerk}} \cdot (a_{\text{steering}, t} - a_{\text{steering}, t-1})^2$$
   * This forces the actor network to output smooth steering profiles without control chatter.
3. **Trajectory Mini-Batch PPO Updates:**
   * Transition from online 1-step A2C to Proximal Policy Optimization (PPO) clipped surrogate loss to improve convergence reliability across training seeds.
