from src.model import *
from src.buffer import ReplayBuffer
import torch
from torch.optim import AdamW
import torch.nn as nn
import numpy as np

class TetrisAgent: 
    def __init__(self, frame_stack: int, ac_dim: int, lr: float = 1e-4, gamma: float = 0.99, max_memory: int = 1000, max_gradient: float = 0.5):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer = ReplayBuffer(max_memory, self.device)
        
        self.model = TetrisAI(frame_stack, ac_dim).to(self.device)
        self.target = TetrisAI(frame_stack, ac_dim).to(self.device)
        
        self.opt = AdamW(params=self.model.parameters(), lr=lr)
        
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.action_dim = ac_dim
        self.max_grad = max_gradient
        
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
            return np.random.randint(self.action_dim)
        
        with torch.no_grad(): 
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = state.to(self.device)
            
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            
            return action
            
    def update(self, batch_size: int):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]
        
        actions = actions.long()
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            max_next_q = self.target(next_states).gather(1, next_actions)           
            
            dones = dones.unsqueeze(-1)
            targets = rewards.unsqueeze(-1) + (1 - dones) * self.gamma * max_next_q
            targets = targets.squeeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q_values, targets)
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.opt.step()
        return loss.item()
    
    def push(self, state, action, reward, next_state, done): 
        self.buffer.push(state, action, reward, next_state, done)
