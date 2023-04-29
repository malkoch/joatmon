import torch
import torch.nn as nn
import torch.nn.functional as F


class TD3Actor(nn.Module):
    def __init__(self, in_features, out_features):
        super(TD3Actor, self).__init__()

        self.hidden1 = nn.Linear(in_features, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_features)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class TD3Critic(nn.Module):
    def __init__(self, in_features, out_features):
        super(TD3Critic, self).__init__()

        self.hidden1 = nn.Linear(in_features + out_features, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value
