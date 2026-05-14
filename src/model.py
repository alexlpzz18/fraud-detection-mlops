import torch
import torch.nn as nn


class FraudDetector(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(FraudDetector, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.red(x)