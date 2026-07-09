import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import model
import sys
import os 

sys.path.append(
    "/home/k981c/Desktop/NewFolder/TCP-Congestion-Control-Using-DQN/"
    "simulation/ns-allinone-3.40/ns-3.40/contrib/opengym/examples/rl-tcp"
)
sys.path.append(
    "/home/k981c/Desktop/NewFolder/TCP-Congestion-Control-Using-DQN/"
    "simulation/ns-allinone-3.40/ns-3.40/contrib/opengym"
)

from State_filter import Agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_LABELS = [
    "relative_time_s",
    "cwnd_bytes",
    "unacked_bytes",
    "ack_packets",
    "avg_rtt_ms",
    "throughput_mbps",
    "lost_packets",
    "jitter_ms",
    "queue_size_bytes",
]
rl_model = model.DeepQNetwork()
rl_model.load_state_dict(torch.load("dqn_model.pt", weights_only=True))

episodes = 1000

agent = model.DQNAgent()
state = agent.send_action(1)[1].to(device)

rtt = []
tp = []

for i in range(episodes):
    action = agent.selectAction(state)
    reward, next_state , metrics = agent.send_action(action)
    next_state = next_state.to(device)
    q_values = agent.policy_net(state)
    action =  torch.argmax(q_values)
    rtt.append(metrics[3].item())
    tp.append(metrics[0].item())

    state = next_state
print(tp)

plt.figure()
plt.plot(rtt, label='RTT')
plt.xlabel('Episode')
plt.ylabel('RTT')
plt.title('RTT')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('reward.png')

plt.figure()
plt.plot(tp, label='Avg Loss', color='orange')
plt.xlabel('Episode')
plt.ylabel('Throughput')
plt.title('Throughput')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss.png')
