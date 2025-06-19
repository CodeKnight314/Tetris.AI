from src.model import *
from src.buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from typing import List

class TetrisAgent: 
    def __init__(self, 
                 frame_stack: int, 
                 ac_dim: int, 
                 lr: float = 1e-4, 
                 gamma: float = 0.99, 
                 max_memory: int = 1000, 
                 max_gradient: float = 0.5, 
                 action_mask: List[int] = [5, 6], 
                 buffer_type: str = "prioritized"):
        
        self.action_mask = action_mask
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if buffer_type == "replay":
            self.buffer = ReplayBuffer(capacity=max_memory, device=self.device)
        elif buffer_type == "prioritized":
            self.buffer = PrioritizedReplayBuffer(capacity=max_memory, alpha=0.6, device=self.device)
        else:
            raise ValueError(f"Invalid buffer type: {buffer_type}. Expected 'replay' or 'prioritized'.")
        
        self.model = TetrisAI(frame_stack, ac_dim).to(self.device)
        self.target = TetrisAI(frame_stack, ac_dim).to(self.device)
        
        self.opt = AdamW(params=self.model.parameters(), lr=lr)
        
        self.criterion = nn.MSELoss(reduction="none")
        self.gamma = gamma
        self.action_dim = ac_dim
        self.max_grad = max_gradient
        self.beta_start = 0.5
        self.beta = self.beta_start
        self.beta_frames = 10000
        self.training_step = 0

        self.update_target_network(True)
        
    def load_weights(self, path: str): 
        self.model.load_weights(path)
        self.target.load_weights(path)
    
    def save_weights(self, path: str):
        self.model.save_weights(path)
        
    def update_target_network(self, hard_update: bool = True, tau: float = 0.05):
        if hard_update: 
            self.target.load_state_dict(self.model.state_dict())
        else: 
            for target_param, param in zip(self.target.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def select_action(self, state: torch.Tensor, epsilon: float = 0.01):
        if np.random.random() < epsilon:
            valid_actions = [i for i in range(self.action_dim) if i not in self.action_mask]
            return np.random.choice(valid_actions)
        
        with torch.no_grad(): 
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = state.to(self.device)
            
            q_values = self.model(state, normalize=False)
            q_values_original = q_values.detach().clone()
            
            if q_values.dim() == 1:
                for action in self.action_mask:
                    q_values[action] = -float("inf")
            else:
                for action in self.action_mask:
                    q_values[0, action] = -float("inf")

            action = torch.argmax(q_values, dim=1).item()
            return (action, q_values_original.sum().item())
            
    def update(self, batch_size: int):
        self.training_step += 1
        
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.beta = min(1.0, self.beta_start + self.training_step * (1.0 - self.beta_start) / self.beta_frames)
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(batch_size, beta=self.beta)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            weights = torch.ones_like(rewards, device=self.device)
            indices = None
        
        states = states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)
        
        with torch.no_grad():
            next_actions = self.model(next_states, normalize=False).argmax(1, keepdim=True)
            max_next_q = self.target(next_states, normalize=False).gather(1, next_actions)           
            
            dones = dones.unsqueeze(-1)
            targets = rewards.unsqueeze(-1) + (1 - dones) * self.gamma * max_next_q
            targets = targets.squeeze(1).to(self.device)

        current_q_values = self.model(states, normalize=False).gather(1, actions.unsqueeze(1)).squeeze(1)
        raw_loss = self.criterion(current_q_values, targets)
        weighted = weights * raw_loss
        loss = weighted.mean() 
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.opt.step()

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            td_errors = targets.detach() - current_q_values.detach()
            new_prios = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.buffer.update_priorities(indices, new_prios)
        
        return loss.item()
    
    def push(self, state, action, reward, next_state, done): 
        self.buffer.push(state, action, reward, next_state, done)