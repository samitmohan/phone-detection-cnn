import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim=51, hidden_dims=[128, 64], output_dim=2):
        super(MLPModel, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
