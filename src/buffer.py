import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, features, next_features):
        self.buffer[self.position] = (state, action, reward, next_state, done, features, next_features)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer[:self.size], batch_size)
        states, actions, rewards, next_states, dones, features, next_features = zip(*batch)

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        features = torch.stack([s.clone().detach() for s in features]).to(self.device)
        next_features = torch.stack([s.clone().detach() for s in next_features]).to(self.device)

        return states, actions, rewards, next_states, dones, features, next_features

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6, device: str = "cuda"):
        super().__init__(capacity, device)
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, features, next_features):
        super().push(state, action, reward, next_state, done, features, next_features)
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
        states, actions, rewards, next_states, dones, features, next_features = zip(*batch)

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        features = torch.stack([s.clone().detach() for s in features]).to(self.device)
        next_features = torch.stack([s.clone().detach() for s in next_features]).to(self.device)

        total = len(self)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, torch.tensor(weights, dtype=torch.float32, device=self.device), indices, features, next_features

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)


class NStepBuffer:
    """Accumulates n-step transitions before pushing to the replay buffer."""
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, features, next_features):
        self.buffer.append((state, action, reward, next_state, done, features, next_features))

        results = []
        if done:
            while len(self.buffer) > 0:
                results.append(self._compute_nstep())
                self.buffer.popleft()
        elif len(self.buffer) == self.n_step:
            results.append(self._compute_nstep())
            self.buffer.popleft()

        return results

    def _compute_nstep(self):
        state, action, _, _, _, features, _ = self.buffer[0]
        _, _, _, next_state, done, _, next_features = self.buffer[-1]

        n_step_return = 0.0
        for i, (_, _, r, _, d, _, _) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                done = True
                _, _, _, next_state, _, _, next_features = self.buffer[i]
                break

        return state, action, n_step_return, next_state, done, features, next_features
