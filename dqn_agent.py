import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        double_dqn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.double_dqn = double_dqn

        self.q = QNetwork(state_dim, action_dim, hidden_dims, dropout).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dims, dropout).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

    def update_target(self) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def act(self, state: torch.Tensor) -> int:
        self.q.eval()
        with torch.no_grad():
            q_values = self.q(state.to(self.device))
            action = int(torch.argmax(q_values).item())
        self.q.train()
        return action