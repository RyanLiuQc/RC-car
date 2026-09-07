# Model Checkpoints and Policy Progression Log

This directory contains trained Reinforcement Learning (RL) policy model weights (`.pth` PyTorch checkpoints). This document records the progression of policy iterations, detailing reward function adjustments, empirical driving behaviors, training stability observations, and targeted improvements across algorithms.

---

## 1. A2C Model Progression and Empirical Observations

### 1.1 `a2c_policy.pth` — A2C Baseline Iteration (First Attempt)

<img src="../docs/media/a2c/a2c_policy_baseline.gif" width="380" alt="A2C Baseline Policy - Wall Hugging" />

* **Algorithm:** Advantage Actor-Critic (A2C, 1-Step TD)
* **Reward Formulation:** Naive baseline speed incentive combined with linear lateral displacement penalties.
* **Observed Driving Behavior:**
  * **Success:** Successfully completes laps without collisions into track boundaries.
  * **Defect:** Exhibits severe **wall-hugging behavior**. The policy settles into a sub-optimal equilibrium where it rides close to outer track boundaries rather than maintaining lane center ($d \approx 0$).

---

### 1.2 `a2c_policy_redesign_1.pth` — A2C Reward Redesign (Iteration 1)

<img src="../docs/media/a2c/a2c_policy_redesign_1.gif" width="380" alt="A2C Redesign 1 - Tight Centering & Steering Jerk" />

* **Algorithm:** Advantage Actor-Critic (A2C, 1-Step TD)
* **Reward Formulation Changes:**
  * **Removal of Arc-Length Progression:** Eliminated linear centerline delta reward ($\Delta s / \Delta t$) to prevent reward exploitation, such as policies oscillating back and forth to maximize reward accumulation.
  * **Gaussian Centering Incentive:** Replaced linear lateral penalty with a steep Gaussian decay term centered at $d=0$:
    $$\text{centering\_factor} = \exp\left(-\left(\frac{d}{\sigma}\right)^2\right)$$
* **Observed Driving Behavior:**
  * **Success:** Excellent lateral centering. The vehicle remains positioned near the track centerline ($d \approx 0$) through straightaways and curves.
  * **Defect:** High control chatter and **steering jerk**. Because an action rate / differential steering penalty ($\|a_t - a_{t-1}\|^2$) was not yet incorporated in A2C, the policy outputs rapid high-frequency steering oscillations.
* **Training Dynamics and Inconsistency:**
  * **High Variance:** Training stability is inconsistent. Repeated training runs under identical hyperparameter settings frequently suffer from performance degradation or policy collapse due to 1-step TD gradient variance.

---

## 2. PPO Model Progression and Empirical Observations

### PPO Baseline Framework & Reward Formulation
When transitioning to Proximal Policy Optimization (PPO, GAE-$\lambda$), several core reward enhancements were established right from the baseline:

* **Removal of Arc-Length Progression:** Eliminated linear centerline delta reward ($\Delta s / \Delta t$) to prevent reward-farming exploits where policies oscillate back and forth to maximize progress terms.
* **Heading Alignment Coupling:** Maintained directionally aligned progress reward $\left( \frac{v}{v_{\text{target}}} \cdot \cos(\theta_{\text{err}}) \right)$.
* **Gaussian Centering Incentive:** Replaced linear lateral displacement penalties with a steep Gaussian decay term centered at $d=0$:
  $$\text{centering\_factor} = \exp\left(-\left(\frac{d}{\sigma}\right)^2\right)$$
* **Differential Action Rate / Steering Jerk Penalty:** Incorporated an action smoothing penalty from the outset of PPO development:
  $$\mathcal{P}_{\text{jerk}} = \gamma_{\text{jerk}} \cdot (a_{\text{steering}, t} - a_{\text{steering}, t-1})^2$$

---

### 2.1 `PPO_policy_exploites_small_d_by_rotating.pth` — PPO Attempt 1 (Rotation Exploit)

<img src="../docs/media/ppo/ppo_policy_rotation_exploit.gif" width="380" alt="PPO Rotation Exploit" />

* **Algorithm:** Proximal Policy Optimization (PPO, GAE-$\lambda$)
* **Reward Formulation Defect:** Included a standalone additive centering term:
  $$\text{total\_reward} = (0.5 \cdot \text{progress\_reward} \cdot \text{centering\_factor}) + (0.5 \cdot \text{centering\_factor}) - \mathcal{P}_{\text{jerk}}$$
* **Observed Driving Behavior:**
  * **Reward Exploit:** The policy discovered a reward-farming exploit where it spins continuously in place near $d \approx 0$. Because the additive `+ 0.5 * centering_factor` term awarded positive continuous returns just for remaining near the centerline without forward velocity, the agent maximized cumulative returns by pivoting in tight circles.

---

### 2.2 `PPO_policy_update_reward.pth` — PPO Attempt 2 (Centerline Lock & Multiplicative Centering)

<img src="../docs/media/ppo/ppo_policy_update_reward.gif" width="380" alt="PPO Update Reward" />

* **Algorithm:** Proximal Policy Optimization (PPO, GAE-$\lambda$)
* **Reward Formulation Changes:**
  * **Removed Additive Centering Factor:** Removed the standalone `+ 0.5 * centering_factor` term, forcing `centering_factor` to act strictly as a multiplicative scalar on velocity progress:
    $$\text{total\_reward} = (\text{progress\_reward} \cdot \text{centering\_factor}) - \mathcal{P}_{\text{jerk}}$$
* **Observed Driving Behavior:**
  * **Eliminated Rotation Exploit:** Forcing the centering term to scale velocity progress multiplicatively required the vehicle to drive forward to receive positive rewards, completely eliminating the spinning exploit.
  * **Major Improvement in Smoothness:** Achieved significant gains in trajectory smoothness and stability compared to earlier reward functions trained without jerk penalties (e.g. A2C).

---

### 2.3 `PPO_policy_update_reward_1.pth` — PPO Attempt 3 (Warm-Started Checkpoint & Smooth Lap Completion)

<img src="../docs/media/ppo/ppo_policy_update_reward_1.gif" width="380" alt="PPO Update Reward 1" />

* **Algorithm:** Proximal Policy Optimization (PPO, GAE-$\lambda$)
* **Training Methodology:**
  * **Warm-Start Checkpoint Training:** Initialized from `PPO_policy_update_reward.pth` (Attempt 2) rather than random weight initialization, allowing the policy to retain learned spatial representations while extending optimization.
* **Observed Driving Behavior:**
  * **Polished Lap Navigation:** Delivers exceptionally smooth, continuous lap navigation with optimal speed control, tight centerline tracking ($d \approx 0$), and zero control chatter.

---

## 3. SAC Model Progression and Empirical Observations

### SAC Baseline Framework & Theoretical Foundations
Soft Actor-Critic (SAC) is an off-policy, actor-critic algorithm based on the **maximum entropy reinforcement learning** framework. Unlike standard policy gradient methods that optimize only expected cumulative return, SAC maximizes both expected return and policy entropy:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$

Key architectural components include:
* **Off-Policy Replay Buffer:** Circular experience storage decoupling data collection from gradient optimization for high sample efficiency.
* **Twin Q-Networks ($Q_1, Q_2$):** Clipped double Q-learning mitigation to prevent overestimation bias:
  $$y = r + \gamma (1 - d) \left( \min(Q_1(s', a'), Q_2(s', a')) - \alpha \log \pi(a' | s') \right)$$
* **Reparameterized Squashed Gaussian Policy:** Outputs continuous actions $a \in [-1, 1]$ via $a = \tanh(u)$ where $u \sim \mathcal{N}(\mu(s), \sigma(s))$, with multivariate Jacobian determinant log-probability correction:
  $$\log \pi(a|s) = \log \mu(u|s) - \sum_{i=1}^{D} \log\left(1 - \tanh^2(u_i) + \epsilon\right)$$
* **Polyak Soft Target Updates:** Continuous tracking of target critic weights:
  $$\theta_{\text{target}} \leftarrow p \cdot \theta_{\text{target}} + (1 - p) \cdot \theta_{\text{current}}, \quad p = 0.995 \ (\tau = 0.005)$$

---

### 3.1 `sac_policy_attempt_1.pth` — SAC Attempt 1 (Detached Actor Gradients)

* **Algorithm:** Soft Actor-Critic (SAC)
* **Implementation Defects:**
  * **Detached Q-Flow:** Q-network evaluations in `loss_actor` were wrapped in `torch.no_grad()`, severing gradient backpropagation to the actor's action output ($\nabla_a Q(s, a) \cdot \nabla_\theta a_\theta(s)$). Therefore, the loss was only tracking the log probability of the action (-entropy), causing the optimizer to only maximize for entropy of the action, without maximizing the policy that maximizes the Q-value.
  * **Replay Buffer Overwrite Bug:** Buffer size was derived from pointer modulo instead of cumulative insertion count, dropping size to 0 upon boundary wrap. 
  * **Standard Deviation Output Dim:** The standard deviation outputted from the actor network did not match the action's dimension causing unintended broadcasting.
* **Observed Driving Behavior:**
  * The policy optimized solely for maximum entropy, producing completely uniform random exploration without ever learning to navigate or track the centerline.

---

### 3.2 `version2/` — SAC Version 2 (Gradient tracking action in Q-value)

<img src="../docs/media/sac/v2/SAC_100000.gif" width="380" alt="SAC Version 2 - 100k Steps" />

* **Algorithm:** Soft Actor-Critic (SAC)
* **Key Enhancements:**
  * **Restored Gradient Flow:** Removed `torch.no_grad()` from the actor loss calculation, enabling policy weights to learn from critic feedback.
  * **Uniform Warmup Exploration:** Initialized first 5,000 steps with `env.action_space.sample()` to seed the replay buffer with diverse states before optimization began.
  * **Periodic Checkpointing:** Implemented intermediate weight serialization every 20k steps.
* **Remaining Defects:**
  * Target networks were updated only intermittently (every 500 steps with $p=0.95$), leading to target lag. The target network was updated too slowly, leading current Q-network chase old and bad target net's weights.
  * Standard deviation parameter `log_std` was unconstrained, risking numerical instability. `log_std` smaller than -20 and bigger than 2 created extreme/exploding standard deviations. 
* **Observed Driving Behavior:**
  * The policy learnt how to not crash into a wall by not moving or reversing to recentralize itself before it hits a wall. It fails to learn better strategy because of lagging target network.

---

### 3.3 `version3/` — SAC Version 3 (Target Updates & Log-Prob Broadcasting Bug)

<img src="../docs/media/sac/v3/SAC_100000.gif" width="380" alt="SAC Version 3 - 100k Steps" />

* **Algorithm:** Soft Actor-Critic (SAC)
* **Key Enhancements:**
  * Independent variance heads for throttle and steering (`log_std` output dimension matching `action_dim = 2`).
  * Stored true `terminated` flag in replay buffer to avoid false zero-masking on episode truncations (a truncated episode can still have a reward at last step. A terminated one can't).
* **Remaining Defects:**
  * **Global Batch Sum Bug:** In `SACActor`, `torch.log(1 - a^2).sum()` reduced across the entire batch of 64 transitions down to a global scalar rather than summing across action dimensions per sample (`dim=1, keepdim=True`).
  * **Broadcasting Distortion:** `log_prob` remained shape `(batch, 2)`, silently broadcasting against `(batch, 1)` Q-values and distorting the Bellman target loss.

---

### 3.4 `version4/` — SAC Version 4 (Full Convergence & High-Speed Lap Optimization)

<img src="../docs/media/sac/v4/SAC_100000.gif" width="380" alt="SAC Version 4 - 100k Steps" />

* **Algorithm:** Soft Actor-Critic (SAC)
* **Key Fixes & Enhancements:**
  * **Correct Multivariate Reduction:** Replaced global sum with `(log_prob_u - torch.log(1 - a^2 + eps)).sum(dim=1, keepdim=True)`, yielding clean `(batch, 1)` scalar log-probabilities with zero tensor broadcasting distortion.
  * **Clamped `log_std`:** Bounded `log_std` within $[-20, 2]$ to prevent variance collapse ($0.0$) and exponential blowup.
  * **Continuous Soft Updates:** Target networks updated every single step (`target_network_update_freq = 1`) with retention $p = 0.995$ ($\tau = 0.005$).
  * **Replay Buffer Sizing:** Reduced buffer capacity to 50,000 transitions for data freshness.
  * **Diagnostic Observability:** Integrated twin critic disagreement ($|Q_1 - Q_2|$), target tracking, and expected policy value to the printed metrics.
* **Observed Driving Behavior & Convergence:**
  * **High-Speed Centerline Navigation:** Achieved smooth, fast lap navigation. The agent discovered that cruising at $\approx 5\text{ m/s}$ along the straight centerline and slowing down on curves maximized returns, driving Q-mean to $\approx 114.1$ with low twin critic disagreement ($0.49$) and stable TD error ($3.42$).

---

## 4. Summary Comparison Matrix

| Model Checkpoint | Algorithm | Primary Feature | Centering ($d \approx 0$) | Speed & Smoothness | Training Stability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`a2c_policy.pth`** | A2C (1-Step) | Linear speed & boundary penalty | Wall Hugging | Low Speed / Oscillations | Inconsistent |
| **`a2c_policy_redesign_1.pth`** | A2C (1-Step) | Gaussian centering & progress coupling | Excellent ($d \approx 0$) | Moderate / Steering Jerk | Inconsistent / High Variance |
| **`PPO_policy_exploites...pth`** | PPO (GAE) | Additive centering reward (`+ 0.5 * centering`) | Rotation Exploit | N/A (Spins in place) | High (Converges on exploit) |
| **`PPO_policy_update_reward.pth`** | PPO (GAE) | Multiplicative centering (removed additive term) | Excellent ($d \approx 0$) | High / Smooth | High |
| **`PPO_policy_update_reward_1.pth`** | PPO (GAE) | Warm-started from Attempt 2 | Excellent ($d \approx 0$) | Optimal / Polished | High / Repeatable |
| **`sac/version2/SAC_100000.pth`** | SAC (Off-Policy) | Restored actor gradients, target lag ($p=0.95$ / 500 steps) | Poor (Avoids walls) | Stationary / Reversing | Inconsistent |
| **`sac/version3/SAC_100000.pth`** | SAC (Off-Policy) | Batch sum bug in log-prob Jacobian | Moderate | Crawling / Slow Progress | Distorted Target Loss |
| **`sac/version4/SAC_100000.pth`** | SAC (Off-Policy) | Max-entropy actor-critic, twin Q-nets, continuous soft updates | Excellent ($d \approx 0$) | High-Speed ($\approx 5.0\text{ m/s}$) & Agile | High / Robust ($|Q_1 - Q_2| \le 0.5$) |

---

## 5. Multi-Track Zero-Shot Generalization (PPO Attempt 3)

To evaluate whether policy representations learned on `default_oval` transfer to unseen track geometries without additional training, `PPO_policy_update_reward_1.pth` (Attempt 3) was evaluated zero-shot on the `s_curve` track layout.

| Track Layout | Simulation Demonstration | Zero-Shot Transfer Analysis |
| :--- | :--- | :--- |
| **`default_oval`** | <img src="../docs/media/ppo/ppo_policy_update_reward_1.gif" width="220" /> | **Primary Training Track:** Achieves continuous lap completion with tight centerline tracking ($d \approx 0$) and optimal speed control. |
| **`s_curve`** | <img src="../docs/media/ppo/ppo_policy_update_reward_1_s_curve.gif" width="220" /> | **Zero-Shot Transfer (Chicane):** Successfully navigates reverse curves and continuous chicanes, maintaining smooth steering adjustments without off-track collisions. |

---

## 6. Targeted Technical Improvements

1. **Frame Stacking / Temporal Memory:**
   * Concatenate past observation frames ($K=4$) to form a 24D state representation, allowing policies to infer lateral velocity $\dot{d}$ and yaw acceleration.
2. **Multi-Track Curriculum Training:**
   * Train policies across randomized track layouts (`default_oval`, `s_curve`, `figure_eight`) to build generalizable trajectory tracking capability.
3. **Automated Entropy Tuning for SAC:**
   * Incorporate automatic dual temperature adjustment ($\alpha$) via stochastic gradient descent:
     $$J(\alpha) = \mathbb{E}_{a \sim \pi_t} \left[ -\alpha \log \pi_t(a|s_t) - \alpha \bar{\mathcal{H}} \right]$$

