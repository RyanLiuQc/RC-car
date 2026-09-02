# """Advantage Actor-Critic (A2C) Agent: on-policy synchronous actor-critic policy gradient algorithm.
#
# This file defines A2CAgent inheriting from BaseAgent. It uses separate ActorNetwork and CriticNetwork
# instances to select continuous actions, estimate baseline state values V(s), and update policy weights
# using 1-step or multi-step Temporal Difference (TD) advantage estimates.
#
# General rule of RL: Max
# input obs -> NN with weights -> output: action
# Therefore: state does NOT depends on NN weights, but policy depends on them 
# since policy is probability distribution of an action IS defined by weights.
# """

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.rl.base_agent import BaseAgent
from src.rl.networks import ActorNetwork, CriticNetwork

class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C) Agent"""

    def __init__(
            self, 
            obs_dim: int = 6, 
            action_dim: int = 2, 
            actor_lr: float = 1e-4, 
            critic_lr: float = 3e-4, 
            gamma: float = 0.99, 
            entropy_coef: float = 0.1,
            device = "cpu"
            ) -> None:
        """Initialize ActorNetwork, CriticNetwork, Adam optimizers, and discount factor gamma."""
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma

        # Instantiate separate Actor and Critic networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(obs_dim)

        # Instantiate Adam optimizers with independent learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.entropy_coef = entropy_coef

        self.device = torch.device("mps") if device == "mps" and torch.backends.mps.is_available() else torch.device(device)


    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select continuous action [throttle, steering] for environment step.
        Returns clipped numpy array of shape (2,).
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_np, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action_np

    def train_step(self, trajectory_buffer: dict, step: int) -> dict:
        """
        Perform 1-step or multi-step Advantage Actor-Critic (A2C) gradient update.
        
        Steps:
        1. Query state_value V(s) and next_state_value V(s_next) from CriticNetwork.
        2. Compute Advantage: A_t = R_t + (gamma * V(s_next) * (1 - terminated)) - V(s)
        3. Evaluate new_log_probs and entropy using self.actor.evaluate_actions(obs, action).
        4. Compute Actor Loss: L_actor = - (new_log_probs * Advantage.detach()) - (c2 * entropy)
        5. Compute Critic Loss: L_critic = MSE(V(s), R_t + gamma * V(s_next))
        6. Zero gradients, perform backpropagation, and step optimizers.

        Input: 
        trajectory_buffer = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs, # after action was taken
            "terminated": terminated # if next_obs hits obstacle or wall
        }
        """
        # Convert all buffer inputs into float32 Tensors (1, N)
        # get curr obs
        obs_np = trajectory_buffer['obs'] # shape (6,)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0) # shape (1,6)

        action_np = trajectory_buffer['action']
        action_tensor = torch.as_tensor(action_np, dtype=torch.float32).unsqueeze(0)

        # get next obs after action was taken
        next_obs_np = trajectory_buffer['next_obs']
        next_obs_tensor = torch.as_tensor(next_obs_np, dtype=torch.float32).unsqueeze(0) 

        # Convert scalar reward & terminated into Tensors
        reward_tensor = torch.as_tensor(trajectory_buffer['reward'], dtype=torch.float32).unsqueeze(0)
        # to mask next_state_value, uses 1-terminated 
        terminated_tensor = torch.as_tensor(trajectory_buffer['terminated'], dtype=torch.float32).unsqueeze(0)

        # query current state value
        state_value = self.critic(obs_tensor) # execute __call__() method instead of forward
        # nn.Module.__call__() does much more than forward()
        with torch.no_grad(): # same as detach
            next_state_value = self.critic(next_obs_tensor) # V(S_{t+1})

        # compute advantage (in this case: 1D Temporal Difference Error)
        # if crash (terminated) after action is taken, then ignore next_state value, put it to 0
        target_state_value = reward_tensor + (self.gamma*next_state_value)*(1.0-terminated_tensor)
        A_t = (target_state_value - state_value).squeeze(-1).detach() # do not compute gradient for this
        # since it is not part of what we want to optimize. We want to optimize log prob (for action)

        # Evaluate new_log_probs and entropy using self.actor.evaluate_actions(obs, action).
        new_log_prob, entropy = self.actor.evaluate_actions(obs_tensor, action_tensor)

        # Compute Actor Loss: L_actor = - (new_log_probs * Advantage.detach()) - (c2 * entropy)
        # entropy parameter added encoourage exploration
        # if action has high entropy 
        L_actor = - (new_log_prob * A_t).mean()  - (self.entropy_coef * entropy.mean())

        # Compute Critic Loss: L_critic = MSE(V(s), R_t + gamma * V(s_next))
        # L_critic = ((state_value - target_state_value)**2).mean()
        L_critic = F.mse_loss(state_value, target_state_value)

        # Zero gradients, perform backpropagation, and step optimizers.
        self.actor_optimizer.zero_grad() # discard weigths from previous train step
        L_actor.backward() # compute gradient of the loss wrt weights
        self.actor_optimizer.step() # update towards min loss (theta = theta - alpha*grad_L)

        self.critic_optimizer.zero_grad()
        L_critic.backward()
        self.critic_optimizer.step()


        return {
            "actor_loss": L_actor.item(),
            "critic_loss": L_critic.item(),
            "entropy": entropy.mean().item(),
            "advantage": A_t.mean().item()
        }

    def save(self, filepath: str) -> None:
        """Save both Actor and Critic network state dicts into a single checkpoint file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load Actor and Critic network state dicts from a checkpoint file."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])