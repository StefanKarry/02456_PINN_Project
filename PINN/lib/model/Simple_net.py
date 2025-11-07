import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, *args, **kwargs):
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding='same'),
        )

        self.fc = nn.Linear(1 * 28 * 28, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x