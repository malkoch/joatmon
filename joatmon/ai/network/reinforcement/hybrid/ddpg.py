import torch
import torch.nn as nn

__all__ = ['DDPGActor', 'DDPGCritic']


class DDPGActor(nn.Module):
    def __init__(self, in_features, out_features):
        super(DDPGActor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=out_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))


class DDPGCritic(nn.Module):
    def __init__(self, in_features, out_features):
        super(DDPGCritic, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(in_features=7 * 7 * 64, out_features=200)
        self.linear2 = nn.Linear(in_features=200 + out_features, out_features=200)
        self.linear3 = nn.Linear(in_features=200, out_features=1)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

    def forward(self, x, y):
        x = self.extractor(x)

        x = self.relu(self.bn1(self.linear1(x.view(x.size(0), -1))))
        x = torch.cat([x, y], dim=1)
        x = self.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)
