import random
import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, features):
        self.buffer[self.position] = (state, action, reward, next_state, done, features)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer[:self.size], batch_size)
        states, actions, rewards, next_states, dones, features = zip(*batch)

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        features = torch.stack([s.clone().detach() for s in features]).to(self.device)

        return states, actions, rewards, next_states, dones, features

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6, device: str = "cuda"):
        super().__init__(capacity, device)
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, features):
        super().push(state, action, reward, next_state, done, features)
        idx = (self.position - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        
        prob_sum = probs.sum()
        if prob_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= prob_sum

        indices = np.random.choice(self.size, batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, features = zip(*batch)

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        features = torch.stack([s.clone().detach() for s in features]).to(self.device)

        total = len(self)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, torch.tensor(weights, dtype=torch.float32, device=self.device), indices, features

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)