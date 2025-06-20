from src.model import *
from src.buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from typing import List
from torch.optim.lr_scheduler import CosineAnnealingLR

class TetrisAgent: 
    def __init__(self, 
                 frame_stack: int, 
                 ac_dim: int, 
                 lr: float = 1e-4, 
                 min_lr: float = 1e-5,
                 gamma: float = 0.99, 
                 max_memory: int = 1000, 
                 max_gradient: float = 0.5, 
                 action_mask: List[int] = [5, 6], 
                 buffer_type: str = "prioritized",
                 scheduler_max: int = 1000000,
                 beta_start: int = 0.5,
                 beta_frames: int = 10000):
        
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
        self.scheduler = CosineAnnealingLR(self.opt, scheduler_max, min_lr)
        
        self.criterion = nn.MSELoss(reduction="none")
        self.gamma = gamma
        self.action_dim = ac_dim
        self.max_grad = max_gradient
        self.beta_start = beta_start
        self.beta = self.beta_start
        self.beta_frames = beta_frames
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
            _, height, width = state.shape
            aggregate_height, bumpiness, max_height = self.calculate_height(state[0])
            holes = self.calculate_holes(state[0])
            features = torch.as_tensor(
                np.array([holes/(height * width), bumpiness / (height * width / 2), aggregate_height / (height * width), max_height / (height)]), 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = state.to(self.device)
            
            q_values = self.model(state, features, normalize=True)
            
            if q_values.dim() == 1:
                for action in self.action_mask:
                    q_values[action] = -float("inf")
            else:
                for action in self.action_mask:
                    q_values[0, action] = -float("inf")

            action = torch.argmax(q_values, dim=1).item()
            return action
            
    def update(self, batch_size: int):
        self.training_step += 1
        
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.beta = min(1.0, self.beta_start + self.training_step * (1.0 - self.beta_start) / self.beta_frames)
            states, actions, rewards, next_states, dones, weights, indices, features = self.buffer.sample(batch_size, beta=self.beta)
        else:
            states, actions, rewards, next_states, dones, features = self.buffer.sample(batch_size)
            weights = torch.ones_like(rewards, device=self.device)
            indices = None
        
        states = states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)
        features = features.float().to(self.device)
        
        with torch.no_grad():
            next_actions = self.model(next_states, features, normalize=True).argmax(1, keepdim=True)
            max_next_q = self.target(next_states, features, normalize=True).gather(1, next_actions)           
            
            dones = dones.unsqueeze(-1)
            targets = rewards.unsqueeze(-1) + (1 - dones) * self.gamma * max_next_q
            targets = targets.squeeze(1).to(self.device)

        current_q_values = self.model(states, features, normalize=True).gather(1, actions.unsqueeze(1)).squeeze(1)
        raw_loss = self.criterion(current_q_values, targets)
        weighted = weights * raw_loss
        loss = weighted.mean() 
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.opt.step()
        self.scheduler.step()

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            td_errors = targets.detach() - current_q_values.detach()
            new_prios = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.buffer.update_priorities(indices, new_prios)
        
        q_value_mean = current_q_values.detach().cpu().numpy().mean() 
        q_value_std = current_q_values.detach().cpu().numpy().std()
        
        return loss.item(), q_value_mean, q_value_std
    
    def push(self, state, action, reward, next_state, done, features): 
        self.buffer.push(state, action, reward, next_state, done, features)
        
    def calculate_holes(self, board: np.array):
        holes = 0
        for col in range(board.shape[1]):
            for row in range(board.shape[0]-1):
                if board[row, col] == 255 and board[row+1, col] == 0 and self.has_path_to_bottom(board, row, col):
                    holes += board.shape[0] - row
        return holes
    
    def has_path_to_bottom(self, board: np.array, start_row: int, start_column: int):
        height, width = board.shape
        visited = set()
        stack = [(start_row, start_column)]
        directions = [(1, 0), (1, -1), (1, 1)]
        
        while stack: 
            row, col = stack.pop()
            
            if row == height: 
                return True
            
            if (row, col) in visited or row < 0 or row >= height or col < 0 or col >= width:
                continue
            
            if board[row, col] != 255: 
                continue 
            
            for dir in directions: 
                delta_y, delta_x = dir 
                new_row = delta_y + row
                new_col = delta_x + col
                if (new_row, new_col) not in visited: 
                    stack.append((new_row, new_col))
                    
        return False
    
    def calculate_height(self, board: np.array):
        heights = np.zeros(board.shape[1])
        for col in range(board.shape[1]):
            highest_row = None
            for row in range(board.shape[0]):
                if board[row, col] == 255:
                    highest_row = row
                    break
            
            if highest_row is None:
                heights[col] = 0
                continue
            
            if self.has_path_to_bottom(board, highest_row, col):
                heights[col] = board.shape[0] - highest_row
            else:
                heights[col] = 0
    
        aggregate_height = sum(heights)
        bumpiness = sum(np.array([abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)]))
        max_height = np.max(heights)
        return aggregate_height, bumpiness, max_height