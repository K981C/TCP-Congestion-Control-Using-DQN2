import torch
import torch.optim as optim
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

batch_size = 64
gamma = 0.5
num_episodes = 1000
steps_per_episode = 100
target_update = 10

memory = model.ReplayBuffer(100000)
agent = model.DQNAgent()

optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
f_result = open("result.txt","w")
f_state = open("state.txt","w")
for episode in range(num_episodes):

    state = agent.send_action(1)[1].to(device)

    episode_reward = 0

    for step in range(steps_per_episode):

        action = agent.selectAction(state)
        reward, next_state = agent.send_action(action)

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

        with torch.no_grad():
            next_q = agent.target_net(next_states)
            max_next_reward = next_q.max(1)[0]
            print(next_q)
            #change
            print(max_next_reward)
            target_q = rewards + gamma * max_next_reward
            
        loss = loss_fn(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()


        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 10)

        optimizer.step()


    if episode % target_update == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())


    agent.epsilon = max(
        agent.epsilon_min,
        agent.epsilon * agent.epsilon_decay
    )

    print(
        f"Episode {episode:3d} | "
        f"Reward {episode_reward:.2f} | "
        f"Epsilon {agent.epsilon:.3f} | "
        f"Loss {loss.item():.4f}"
    )
   
    
    f_result.write(
    f"Episode {episode:3d} | "
    f"Reward {episode_reward:.2f} | "
    f"Epsilon {agent.epsilon:.3f} | "
    f"Loss {loss.item():.4f}"
    f"\n"
    )  
    f_result.flush()