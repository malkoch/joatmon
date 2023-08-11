import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

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
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))
