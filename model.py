import torch
import torch.nn as nn


class PendulumNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, t):
        return self.network(t)


if __name__ == "__main__":
    model = PendulumNet()

    # Test with 5 time values
    t = torch.linspace(0, 12, 5).reshape(-1, 1)

    theta_pred = model(t)

    print("Input shape:", t.shape)
    print("Output shape:", theta_pred.shape)
    print("Predictions:", theta_pred)
