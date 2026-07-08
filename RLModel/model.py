import random
import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(
    "/home/k981c/Desktop/NewFolder/TCP-Congestion-Control-Using-DQN/"
    "simulation/ns-allinone-3.40/ns-3.40/contrib/opengym/examples/rl-tcp"
)
sys.path.append(
    "/home/k981c/Desktop/NewFolder/TCP-Congestion-Control-Using-DQN/"
    "simulation/ns-allinone-3.40/ns-3.40/contrib/opengym"
)

from State_filter import Agent

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 81),
            nn.ReLU(),
            nn.Linear(81, 162),
            nn.ReLU(),
            nn.Linear(162 , 81),
            nn.ReLU(),
            nn.Linear(81, 6)
        )
        
    def forward(self, x):
        return self.network(x)
    
    


class ReplayBuffer:
    """Stores and samples environment transitions to break correlation."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
            
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
            
    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.stack(state).float().to(device),
            torch.LongTensor(action).to(device),
            torch.stack(reward).float().to(device),
            torch.stack(state).float().to(device),
        
        )
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    epsilon_min = 0.1
    epsilon_decay = 0.995
    def __init__(self):
        self.policy_net = DeepQNetwork().to(device)
        self.target_net =  DeepQNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=0.001)  
        self.epsilon = 1.0
        self.alpha = 0.78
     

    def selectAction(self , state):
        global epsilon
        if random.uniform(0.0,1.0) < self.epsilon:
            return random.randint(0,5)
        else:
            with torch.no_grad():
                q = self.policy_net(state)
            return q.argmax().item()
    def send_action(self ,action):
        results = Agent(action)

        T = results[1][0]
        Tmin = results[1][1]
        Tmax = results[1][2]
        rtt = results[1][3]
        rttmin = results[1][4]
        rttmax = results[1][5]
        # print("T=",T,"| Tmin=",Tmin,"| Tmax=",Tmax,"| rtt=",rtt,"| rttmin=",rttmin,"| rttmin=",rttmax)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_T = (T - Tmin) / (Tmax - Tmin)
            norm_rtt = (rtt - rttmin) / (rttmax - rttmin)
            # Replace any NaNs resulting from 0/0 with 0.0
            norm_T = np.nan_to_num(norm_T)
            norm_rtt = np.nan_to_num(norm_rtt)
            r_t = self.alpha * norm_T - (1 - self.alpha) * norm_rtt
            state = torch.as_tensor(results[0])
           
        return r_t,state
    
    def store_action_and_reward():
        memory = ReplayBuffer(100000)
        memory.push()
