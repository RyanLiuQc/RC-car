# """Soft Actor-Critic (SAC) Agent: off-policy maximum-entropy actor-critic algorithm.
#
# This file implements SACAgent, inheriting from BaseAgent. It uses an off-policy
# ReplayBuffer, twin Q-networks, and entropy maximization to optimize sample-efficient
# continuous vehicle control.
# """

import numpy as np
from src.rl.base_agent import BaseAgent
from src.rl.replay_buffer import ReplayBuffer
from src.rl.networks import SACActor, SACCritic

import torch
from torch.nn import utils
from torch import optim
import torch.nn.functional as F


class SACAgent(BaseAgent):
    def __init__(
            self, 
            obs_dim: int = 6, 
            action_dim: int = 2, 
            actor_lr: float = 1e-4, 
            critic_lr: float = 3e-4,
            gamma: float = 0.99,
            buffer_capacity: int = 100000,
            # n_epochs: int = 10, # we don't go through ALL element of dataset... 
            # instead, we just pick randomly sample 32 batches of 64-step (batch_size) (still 2048 steps processed)
            batch_size: int = 64,
            entropy_coef: float = 0.01,
            start_step: int = 5000, # number of steps done to collect data without training before starting to optimize at every step
            target_network_update_freq: int = 500
            ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        # self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.start_step = start_step
        self.target_network_update_freq = target_network_update_freq

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.actor_lr: float = actor_lr
        self.critic_lr: float = critic_lr
        self.gamma = gamma

        self.Q1 = SACCritic(obs_dim, action_dim).to(self.device)
        self.Q2 = SACCritic(obs_dim, action_dim).to(self.device)

        # COPY the parameters from Q1 and same thing for Q2
        self.Q1_target = SACCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.Q2_target = SACCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.actor = SACActor(obs_dim=obs_dim, action_dim=action_dim).to(self.device)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim, action_dim=action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # since q1 and q2 weights come from different branches of the computational graph, 
        # they won't affect each other during backward computation.
        self.Q_optimizer = torch.optim.Adam(list(self.Q1.parameters())+list(self.Q2.parameters()), lr=critic_lr)

        # self._count = 0 # keep track of training steps done
        

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select continuous action using reparameterized PolicyNetwork."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # (1,6)
        if deterministic:
            # maybe need to detach (no grad to make this work before converting to numpy)
            # tanh to squash action btw -1 and 1 like what sample_action() does. actor() only outputs unsquashed.
            return torch.tanh(self.actor(obs_tensor)[0]).detach().cpu().numpy().squeeze(0)

        action, log_prob_a = self.actor.sample_action(obs_tensor)

        return action.detach().cpu().numpy().squeeze(0)

    def train_step(self, trajectory_buffer: dict, step: int) -> dict:
        """Perform SAC soft Q-learning update step over mini-batch.
        Input: trajectory_buffer
        Output: metrics = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies)
        }
        """
        # TODO: Implement soft Bellman backup and entropy minimization updates.
        # RECALL
        # trajectory_buffer: dict = {
        #     "obs": obs,
        #     "action": action,
        #     "reward": reward,
        #     "next_obs": next_obs, 
        #     "terminated": terminated,
        #     "done": done # where done = terminated or truncated
        # }

        # self._count += 1

        # retrieve data
        obs: np.ndarray = trajectory_buffer["obs"] 
        action: np.ndarray = trajectory_buffer["action"]
        reward: float = trajectory_buffer["reward"]
        next_obs: np.ndarray = trajectory_buffer["next_obs"]
        terminated: bool = trajectory_buffer["terminated"] # we should mask ONLY when episode ends
        # episode ends -> no next step -> no reward
        # truncated -> previous steps where smooth -> next step could still exist -> reward can exist (we just stopped recording)

        # add to replay buffer
        self.replay_buffer.add(obs,action,reward,next_obs,terminated)

        
        if step < self.start_step:
            
            return {}

        sample = self.replay_buffer.sample()
        obs_batch = torch.from_numpy(sample["observations"]).to(self.device)
        action_batch = torch.from_numpy(sample["actions"]).to(self.device)
        reward_batch = torch.from_numpy(sample["rewards"]).to(self.device)
        next_obs_batch = torch.from_numpy(sample["next_obs"]).to(self.device)
        terminated_batch = torch.from_numpy(sample["terminated"]).to(self.device)

        assert list(next_obs_batch.shape) == [self.batch_size, self.obs_dim], f"next_obs_batch.shape should be {[self.batch_size, self.obs_dim]}, not {list(next_obs_batch.shape)}"
        assert list(obs_batch.shape) == [self.batch_size, self.obs_dim], f"obs_batch.shape should be {[self.batch_size, self.obs_dim]}, not {list(obs_batch.shape)}"
        assert list(action_batch.shape) == [self.batch_size, self.action_dim], f"action_batch.shape should be {[self.batch_size, self.action_dim]}, not {list[action_batch.shape]}"
        assert list(reward_batch.shape) == [self.batch_size, 1], f"reward_batch.shape should be {[self.batch_size, 1]}, not {reward_batch.shape}"
        assert list(terminated_batch.shape) == [self.batch_size, 1], f"done_batch.shape should be {[self.batch_size, 1]}, not {done_batch.shape}"


        # sample and update at every step above 5000 (add step argument to train_step)
        with torch.no_grad():
            # -------------- for Q update -----------------
            # get next_action with current policy
            next_action_batch, log_prob_next_action_batch = self.actor.sample_action(next_obs_batch)
            
            # compute Q(next_state, next_action)
            next_Q1 = self.Q1_target(next_obs_batch, next_action_batch)
            next_Q2 = self.Q2_target(next_obs_batch, next_action_batch)

            # compute target
            Q_targets = (reward_batch + (1.0-terminated_batch.float())*self.gamma * (torch.min(next_Q1, next_Q2) - self.entropy_coef * log_prob_next_action_batch))

            

        # ---------------- for Q update -------------------
        # infer Qs
        Q1_curr = self.Q1(obs_batch, action_batch)
        Q2_curr = self.Q2(obs_batch, action_batch)

        loss_q1 = ((Q1_curr - Q_targets)**2).mean()
        loss_q2 = ((Q2_curr - Q_targets)**2).mean()

        loss_q = loss_q1 + loss_q2

        # -------------- for policy update ----------------
        pred_action_batch, pred_log_prob_action_batch = self.actor.sample_action(obs_batch)

        # DO NOT torch.no_grad, since action prediction flows 
        # into input of Q and we need to learn parameters for action too.
        # draw out the computational graph to visualize
        # we will not learn parameters for the Q-network here 
        # since actor_optimizer only tracks self.actor.parameters()
        # so loss_actor.backward() gradient is computed wrt actor parameters
        # Q1, Q2 parameters are treated as constants.
        Q1_with_action_from_curr_policy = self.Q1(obs_batch, pred_action_batch)
        Q2_with_action_from_curr_policy = self.Q2(obs_batch, pred_action_batch)

        loss_actor = - (torch.min(Q1_with_action_from_curr_policy, Q2_with_action_from_curr_policy) - self.entropy_coef * pred_log_prob_action_batch).mean()

        # actor needs to be computed first since it uses current Q-nets
        # if Q-nets gets updated before actor, we will be flowing back new weights which crash the code
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        self.Q_optimizer.zero_grad()
        loss_q.backward()
        self.Q_optimizer.step()

        if step % self.target_network_update_freq == 0:
            self.update_target_net(self.Q1, self.Q2, self.Q1_target, self.Q2_target)

        
            

        return {
            "actor_loss": loss_actor.item(),
            "critic_loss": loss_q.item(),
            "entropy": pred_log_prob_action_batch.detach().mean().item()
        }

    def update_target_net(self, Q1: SACCritic, Q2: SACCritic, Q1_target: SACCritic, Q2_target: SACCritic) -> None:
        """Update target by tau=5% (ie. keep 95% of target param (polyak p retention rate) and move 5% towards curr param)"""
        q1_old_params = Q1_target.parameters()
        q2_old_params = Q2_target.parameters()

        q1_new_params = Q1.parameters()
        q2_new_params = Q2.parameters()

        # retention rate
        p = 0.95

        # update target parameters (old params)
        for q1_old_param, q2_old_param, q1_new_param, q2_new_param in zip(
            q1_old_params,q2_old_params,q1_new_params,q2_new_params
            ):
            # modify in-place paramater with copy_() ("_" means modify in-place in pytorch)
            q1_old_param.data.copy_(
                p * q1_old_param.data + (1.0 - p) * q1_new_param.data
            )

            q2_old_param.data.copy_(
                p * q2_old_param.data + (1.0 - p) * q2_new_param.data
            )


         

    def save(self, filepath: str) -> None:
        """Save SAC actor and twin critic weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.Q1.state_dict(),
            'q2_state_dict': self.Q2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.Q_optimizer.state_dict()
        },
        filepath
        )

    def load(self, filepath: str) -> None:
        """Load SAC actor and twin critic weights from disk."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.Q1.load_state_dict(checkpoint['q1_state_dict'])
        self.Q2.load_state_dict(checkpoint['q2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])
