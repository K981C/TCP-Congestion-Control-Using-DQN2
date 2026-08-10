import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class DeepQNetwork(nn.Module):
    """Simple MLP that maps the 9-feature ACK state to six Q-values."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128 , 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Keep the existing replay-buffer interface and behavior."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.stack(state).float().to(device),
            torch.LongTensor(action).to(device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.stack(next_state).float().to(device),
            torch.FloatTensor(done).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double DQN with Polyak target updates and stable reward shaping."""

    def __init__(self):
        self.policy_net = DeepQNetwork().to(device)
        self.target_net = DeepQNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.7
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 400000.0
        self.epsilon = self.epsilon_start
        self.train_steps = 0
        self.beta = 0.2



    def selectAction(self, state):
        state = state.to(device)
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()




    def decay_epsilon(self):
        """Use exponential decay to reduce exploration over time."""
        if self.train_steps <= 0:
            self.epsilon = self.epsilon_start
            return
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** (self.train_steps / self.epsilon_decay),
        )
        # print(self.epsilon)


    def soft_update_target_network(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    def get_reward(self, results):
        T = results[0]          # Throughput (Mbps)
        rtt = results[3]        # Avg RTT (ms)
        lost = results[6]       # Lost packets (we will add this in step 2)
        # Reward shaping (scaled to roughly [-1, 1])
        target_tp = 2.0
        target_rtt = 40.0       # expected baseline RTT in ms

        tp_reward = float(np.clip(T / target_tp, 0.0, 1.0))
        # RTT penalty normalized to [0,1]
        rtt_penalty = float(np.clip((rtt - target_rtt) / target_rtt, 0.0, 1.0))

        # lost is a per-step flag (0 or 1); make it a stronger but bounded penalty
        loss_penalty = -1.0 * float(lost)
        stall_penalty = -0.5 if T < 0.1 else 0.0

        reward = (0.3 * tp_reward) - (0.99 * rtt_penalty) + loss_penalty + stall_penalty
        # Clip reward to avoid exploding magnitudes
        reward = float(np.clip(reward, -1.0, 1.0))
        reward = reward /10
        return torch.tensor(reward, dtype=torch.float32, device=device)