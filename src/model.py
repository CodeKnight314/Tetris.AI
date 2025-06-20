import torch
import torch.nn as nn 

class TetrisAI(nn.Module):
    def __init__(self, num_frames: int, action_space: int, num_features: int = 4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4, padding=2), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        
        feature_size = 64 * 7 * 7
        
        self.feature_mcp = nn.Sequential(
            nn.Linear(num_features, 32), 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size + 16, 512), 
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size + 16, 512), 
            nn.ReLU(), 
            nn.Linear(512, action_space)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x: torch.Tensor, features: torch.Tensor, normalize: bool = False):
        if normalize: 
            x = x / 255.0
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        board_embeddings = self.cnn(x)
        feature_embeddings = self.feature_mcp(features)
        
        combined = torch.cat([board_embeddings, feature_embeddings], dim=1)
        
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        
        Q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q_values
    
    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        
    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)