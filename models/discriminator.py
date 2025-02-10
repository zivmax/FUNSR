import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, geometric_init=True, bias=0.5, out_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(1, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, 1)

        if geometric_init:
            for fc in [self.fc1, self.fc2, self.fc3]:
                nn.init.constant_(fc.bias, -bias)
                nn.init.normal_(fc.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            nn.init.normal_(
                self.fc4.weight, mean=np.sqrt(np.pi) / np.sqrt(1), std=0.0001
            )
            nn.init.constant_(self.fc4.bias, -bias)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

    def sdf(self, x):
        return self.forward(x)
