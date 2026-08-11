# """Proximal Policy Optimization (PPO) Agent: on-policy actor-critic algorithm.
#
# This file implements PPOAgent, inheriting from BaseAgent. It optimizes a stochastic
# policy network using a clipped surrogate loss objective and Generalized Advantage
# Estimation (GAE) to deliver stable continuous vehicle control.
# """

import numpy as np
from src.rl.base_agent import BaseAgent
from src.rl.networks.actor import ActorNetwork
from src.rl.networks.critic import CriticNetwork
from src.rl.rollout_buffer import RolloutBuffer

import torch
from torch.nn import utils
from torch import optim
import torch.nn.functional as F



class PPOAgent(BaseAgent):
    def __init__(
            self, 
            obs_dim: int = 6,
            action_dim: int = 2, 
            actor_lr: float = 1e-4, 
            critic_lr: float = 3e-4,
            gamma: float = 0.99, # discount factor of the return
            gae_lambda: float = 0.95, # variance knob for future reward (TD decay factor)
            clip_epsilon: float = 0.20, # PPO loss function's clip percentage value
            n_epochs: int = 10, # number of passes over ALL the dataset (retraining again by randomizing again 10 times)
            batch_size: int = 64, # seperate dataset into batches of 64 random steps collected
            rollout_size: int = 2048, # size of trajectory dataset
            entropy_coef: float = 0.01, # reward uncertainty (maximizing J+H, might also decrease pi(a|s) for more uncertainty)
            max_grad_norm: float = 0.5, # for gradient clipping
            device: str = "cpu"
            ) -> None:
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("mps") if device == "mps" and torch.backends.mps.is_available() else torch.device(device)

        # Networks & Optimizers
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Internal Rollout Memory
        self.rollout_buffer = RolloutBuffer(
            buffer_size=rollout_size, 
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            device=device)
        
        # Transient caches for current step metadata. Used by PPO Actor Loss function
        self._last_log_prob = 0.0
        self._last_value = 0.0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select continuous action using Actor network's Gaussian distribution with a forward pass.
        Cache log_prob and state value internally for train_step
        """
        # old_log_probs is initialized by rollout_buffer to be 0.0 
        # so we need to do forward pass and set self._last_log_prob as our first log_prob

        # unsqueeze (6,) to (1,6)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_np, log_prob = self.actor.get_action(obs_tensor,deterministic=deterministic)
            value = self.critic(obs_tensor).squeeze(-1).item()

        self._last_log_prob = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
        self._last_value = value

        return action_np

    def train_step(self, trajectory_buffer: dict) -> dict:
        """Perform PPO clipped policy gradient update step.
        While buffer is not full, gather data by adding to buffer.
        Once buffer reach capacity:
            calls _ppo_update which is an optimization step over sample mean of clip(ratio_t*A_t)

        """
        # RECALL
        # trajectory_buffer: dict = {
        #     "obs": obs,
        #     "action": action,
        #     "reward": reward,
        #     "next_obs": next_obs, 
        #     "terminated": terminated
        # }

        obs = trajectory_buffer["obs"]
        action = trajectory_buffer['action']
        reward = trajectory_buffer['reward']
        next_obs = trajectory_buffer['next_obs']
        # terminated = trajectory_buffer['terminated']
        done = trajectory_buffer["done"]

        # collect set of data
        self.rollout_buffer.add(
            obs=obs,
            action=action,
            log_prob=self._last_log_prob,
            reward=reward,
            value=self._last_value,
            done=done
        )

        # if rollout buffer not full, just return without optimizing
        if not self.rollout_buffer.is_full():
            return {}
        
        # since buffer full, compute gaea and optimize
        with torch.no_grad():
            # shape = (1,6)
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).squeeze(0) 

            last_value = self.critic(next_obs_tensor).squeeze(-1).item()

        self.rollout_buffer.compute_gae(last_value, done, gamma=self.gamma, gae_lambda=self.gae_lambda)

        metric = self._ppo_update()
        self.rollout_buffer.reset()
        return metric
    
    def _ppo_update(self):
        """Perform SGD updates over rollout mini-batches for K epochs."""
        actor_losses, critic_losses, entropies = [],[],[]
        
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_batches(batch_size=self.batch_size):
                obs_b, actions_b = batch["obs"], batch["actions"]
                old_log_probs_b, adv_b= batch['old_log_probs'], batch["advantages"]
                rtns_b, values_b = batch["returns"], batch["values"]

                # normalize advantages vector for SGD
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # evaluate currect prob and and entropy
                # first batch will yield exact same log_prob since no optimization step happened yet
                # returns tensor (batch_size,)
                new_log_probs, entropy = self.actor.evaluate_actions(obs_batch=obs_b, action_batch=actions_b)

                current_values = self.critic(obs_b).squeeze(-1)

                # Ratio and Clipped Objective
                ratios = torch.exp(new_log_probs - old_log_probs_b)
                loss1 = ratios * adv_b
                clipped_loss2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * adv_b

                # add - to maximize
                actor_loss = -torch.min(loss1, clipped_loss2).mean() - self.entropy_coef * entropy.mean()

                critic_loss = F.mse_loss(current_values, rtns_b)

                # optmize
                self.actor.zero_grad()
                actor_loss.backward()
                # clip gradient change
                utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic.zero_grad()
                critic_loss.backward()
                utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        return {
            actor_loss: np.mean(actor_losses),
            critic_loss: np.mean(critic_losses),
            entropy: np.mean(entropies)
        }




    def save(self, filepath: str) -> None:
        """Save PPO actor and critic weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)


    def load(self, filepath: str) -> None:
        """Load PPO actor and critic weights from disk."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])