import random
import torch

class ReplayBuffer:
    def __init__(self, max: int, device: str = "cuda"):
        self.max = max
        self.device = device
        self.buffer = [None] * max
        self.position = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done):
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max
        self.size = min(self.size + 1, self.max)
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer[:self.size], batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return self.size
    