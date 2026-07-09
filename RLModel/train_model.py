import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import model
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

batch_size = 64
gamma = 0.96
num_episodes = 1000
steps_per_episode = 100
target_update = 10

memory = model.ReplayBuffer(200000)
agent = model.DQNAgent()
state = agent.send_action(1)[1].to(device)
lr = 1e-3
optimizer = optim.Adam(agent.policy_net.parameters(), lr)
loss_fn = torch.nn.MSELoss()
f_result = open("result.txt","w")
f_state = open("state.txt","w")
episode_rewards = []
episode_losses = []
for episode in range(num_episodes):


    episode_reward = 0.0
    episode_loss = 0.0
    train_steps = 0

    for step in range(steps_per_episode):

        
        action = agent.selectAction(state)
        reward, next_state , metrics = agent.send_action(action)

        next_state = next_state.to(device)

        reward = torch.tensor(reward,
                              dtype=torch.float32,
                              device=device)
        
        memory.push(
            state,
            action,
            reward,
            next_state
        )

        episode_reward += reward.item()
  
        state = next_state
      
        # f_state.write(
        # str(state)
        # )
        # f_state.write("\n")
        # f_state.flush()
    
    
        if len(memory) < batch_size:
            continue

        states, actions, rewards, next_states = memory.sample(batch_size)

        # Save transition
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        current_q = agent.policy_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # print(current_q)
        with torch.no_grad():
            next_q = agent.target_net(next_states)
            max_next_reward = next_q.max(1)[0]
            target_q = rewards + gamma * max_next_reward
            # print("target", target_q)
        loss = loss_fn(current_q, target_q)
        # print("loss",loss)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)

        optimizer.step()

        episode_loss += loss.item()
        train_steps += 1
        print("[State]")
        for j, label in enumerate(STATE_LABELS):
            print(f"    {label:30s} = {state[j]:.4f}")
        print()

    if episode % target_update == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(memory.__len__())

    agent.epsilon = max(
        agent.epsilon_min,
        agent.epsilon * agent.epsilon_decay
    )

    avg_loss = episode_loss / train_steps if train_steps > 0 else 0.0
    episode_rewards.append(episode_reward)
    if(num_episodes > 1):
        episode_losses.append(avg_loss)

    print(
        f"Episode {episode:3d} | "
        f"Reward {episode_reward:.2f} | "
        f"Avg Loss {avg_loss:.4f} | "
        f"Epsilon {agent.epsilon:.3f}"
    )
   
   
    f_result.write(
    f"Episode {episode:3d} | "
    f"Reward {episode_reward:.2f} | "
    f"Avg Loss {avg_loss:.4f} | "
    f"Epsilon {agent.epsilon:.3f}"
    f"\n"
    )  
    f_result.flush()
torch.save(agent.policy_net.state_dict(), 'dqn_model.pt')


# plt.figure()
# plt.plot(episode_rewards, label='Reward')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Episode Reward')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig('reward.png')

# plt.figure()
# plt.plot(episode_losses, label='Avg Loss', color='orange')
# plt.xlabel('Episode')
# plt.ylabel('Loss')
# plt.title('Episode Average Loss')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig('loss.png')