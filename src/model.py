import torch
import torch.nn as nn 

class TetrisAI(nn.Module):
    def __init__(self, num_frames: int, action_space: int):
        super().__init__()
        
        self.features = nn.Sequential(*[
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        ])
        
        self.value_stream = nn.Sequential(*[
            nn.Linear(64 * 7 * 7, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        ])
        
        self.advantage_stream = nn.Sequential(*[
            nn.Linear(64 * 7 * 7, 256), 
            nn.ReLU(), 
            nn.Linear(256, action_space)
        ])
        
    def forward(self, x: torch.Tensor, normalize: bool = False):
        if normalize: 
            x = x / 255.0
        main_features = self.features(x)
        value = self.value_stream(main_features)
        advantage = self.advantage_stream(main_features)
        Q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q_values
    
    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path))
        
    def save_weights(self, path: str):
        torch.save(self.state_dict, path)