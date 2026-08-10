#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import os
import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from ns3gym import ns3env
import model
import state_filter
from actions import RLTCP



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# ---- Clipping constants ----
BOTTLENECK_MBPS = 2.0          # your bottleneck bandwidth
MAX_REASONABLE_MBPS = 2.5      # allow a little overhead


tcp_agent = RLTCP()




def run_episode(env, tcp_agent, agent, filter_obj, replay_buffer, train=True, debug=False):
    """Run one episode and return collected data + total reward."""
    obs = env.reset()
    filter_obj.reset()
    state, metrics = filter_obj.extract(obs)

    episode_data = {
        'throughput': [],
        'rtt_ms': [],
        'cwnd': [],
        'rewards': [],
        'actions': [],
        'timestamps': [],
        'loss_events': []
    }
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        # ---- 1. Select action (int 0..5) ----
       

        # Inside run_episode, BEFORE the loop:

        # Inside the loop:
        action_all = tcp_agent.get_action(state, obs, agent)  # Pass state and obs
        action_idx = int(action_all[2])
        actions = [int(action_all[0]), int(action_all[1])]  # [new_ssThresh, new_cwnd]
        # Clip to C++ action space limits (uint16-ish in C++ example: 0..65535)
        actions = np.clip(actions, 0, 65535)
        actions = actions.astype(int)

        if debug and step < 5:
            print(f"[DEBUG] step={step}, calledFunc={obs[11]}, action={actions}, action_idx={action_idx}, obs_cwnd={obs[5]}, obs_ssThresh={obs[4]}")
   





        # ---- 3. Step environment ----
        obs, env_reward, done, info = env.step(actions)
        # print(f"Simulation time: {info}")

        next_state, metrics = filter_obj.extract(obs)




        # ---- 4. Compute reward using agent.get_reward() ----
        reward_tensor = agent.get_reward(metrics)   # scalar tensor
        reward_scalar = reward_tensor.item()





        # ---- 5. Store transition ----
        replay_buffer.push(state, action_idx, reward_scalar, next_state, done)





        # ---- 6. Training (if enough samples) ----
        if train and len(replay_buffer) >= 64:
            state_b, action_b, reward_b, next_b, done_b = replay_buffer.sample(64)




            with torch.no_grad():
                next_actions = agent.policy_net(next_b).argmax(1, keepdim=True)
                next_q = agent.target_net(next_b).gather(1, next_actions).squeeze()
                target = reward_b + agent.gamma * next_q * (1 - done_b)




            current_q = agent.policy_net(state_b).gather(1, action_b.unsqueeze(1)).squeeze()
            loss = torch.nn.functional.mse_loss(current_q, target)





            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()




            agent.train_steps += 1
            agent.decay_epsilon()



            agent.soft_update_target_network()





        # ---- 7. Collect metrics (with clipping) ----
        t = obs[2] / 1e6
        # Clip throughput to a reasonable range (bottleneck is 2 Mbps)
        tp = metrics[0]
        tp_clipped = np.clip(tp, 0.0, MAX_REASONABLE_MBPS)
        episode_data['timestamps'].append(t)
        episode_data['throughput'].append(tp_clipped)          # clipped
        episode_data['rtt_ms'].append(obs[9] / 1000.0)
        episode_data['cwnd'].append(obs[5])
        episode_data['rewards'].append(reward_scalar)
        episode_data['actions'].append(action_idx)
        total_reward += reward_scalar
        step += 1




        # Detect loss (cwnd reduction)
        if len(episode_data['cwnd']) > 1:
            if episode_data['cwnd'][-1] < episode_data['cwnd'][-2]:
                episode_data['loss_events'].append(step - 1)

        state = next_state

    return episode_data, total_reward




def plot_results(episode_data_list, save_dir='results', title='Performance'):
    """Generate six plots with correct time-averaging using interpolation."""
    os.makedirs(save_dir, exist_ok=True)
    num_ep = len(episode_data_list)
    if num_ep == 0:
        return

    # Create a common time axis (from 0 to max episode time, 500 points)
    max_time = np.max([d['timestamps'][-1] for d in episode_data_list])
    common_t = np.linspace(0, max_time, 500)

    avg_tp = np.zeros(len(common_t))
    avg_rtt = np.zeros(len(common_t))
    avg_cw = np.zeros(len(common_t))
    avg_rw = np.zeros(len(common_t))

    for d in episode_data_list:

        # Interpolate each episode to the common time axis
        avg_tp += np.interp(common_t, d['timestamps'], d['throughput']) / num_ep
        avg_rtt += np.interp(common_t, d['timestamps'], d['rtt_ms']) / num_ep
        avg_cw += np.interp(common_t, d['timestamps'], d['cwnd']) / num_ep
        avg_rw += np.interp(common_t, d['timestamps'], d['rewards']) / num_ep

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    # 1) Throughput
    ax = axes[0, 0]
    for d in episode_data_list:
        ax.plot(d['timestamps'], d['throughput'], alpha=0.2, color='blue')
    ax.plot(common_t, avg_tp, 'r-', lw=2, label='Avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.grid(alpha=0.3)
    ax.legend()

    # 2) RTT
    ax = axes[0, 1]
    for d in episode_data_list:
        ax.plot(d['timestamps'], d['rtt_ms'], alpha=0.2, color='green')
    ax.plot(common_t, avg_rtt, 'r-', lw=2, label='Avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RTT (ms)')
    ax.grid(alpha=0.3)
    ax.legend()

    # 3) CWND
    ax = axes[1, 0]
    for d in episode_data_list:
        ax.plot(d['timestamps'], d['cwnd'], alpha=0.2, color='purple')
    ax.plot(common_t, avg_cw, 'r-', lw=2, label='Avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CWND (bytes)')
    ax.grid(alpha=0.3)
    ax.legend()

    # 4) Reward
    ax = axes[1, 1]
    for d in episode_data_list:
        ax.plot(d['timestamps'], d['rewards'], alpha=0.2, color='orange')
    ax.plot(common_t, avg_rw, 'r-', lw=2, label='Avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.grid(alpha=0.3)
    ax.legend()

    # 5) Action Distribution (Updated labels for new action space)
    ax = axes[2, 0]
    all_actions = np.concatenate([d['actions'] for d in episode_data_list])
    counts = np.bincount(all_actions, minlength=6)
    labels = ['x2', '+340', '+1500', 'stay', 'x0.2', 'x0.7']
    ax.bar(range(6), counts, color=plt.cm.Set3(np.linspace(0, 1, 6)))
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title('Action Distribution')

    # 6) Loss events
    ax = axes[2, 1]
    losses = []
    for d in episode_data_list:
        losses.extend([d['timestamps'][i] for i in d['loss_events'] if i < len(d['timestamps'])])
    if losses:
        ax.hist(losses, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Loss Events')
    else:
        ax.text(0.5, 0.5, 'No loss events', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Loss Events')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots.png'), dpi=300)
    plt.show()

    # Summary statistics (using clipped data)
    all_tp = np.concatenate([d['throughput'] for d in episode_data_list])
    all_rtt = np.concatenate([d['rtt_ms'] for d in episode_data_list])
    all_cw = np.concatenate([d['cwnd'] for d in episode_data_list])
    all_rw = np.concatenate([d['rewards'] for d in episode_data_list])
    print("\n=== Summary Statistics (clipped throughput) ===")
    print(f"Throughput: {np.mean(all_tp):.2f} ± {np.std(all_tp):.2f} Mbps")
    print(f"RTT:        {np.mean(all_rtt):.2f} ± {np.std(all_rtt):.2f} ms")
    print(f"CWND:       {np.mean(all_cw):.2f} ± {np.std(all_cw):.2f} bytes")
    print(f"Reward:     {np.mean(all_rw):.3f} ± {np.std(all_rw):.3f}")
    total_loss = sum(len(d['loss_events']) for d in episode_data_list)
    print(f"Loss events: {total_loss}")
    most_common = np.argmax(np.bincount(all_actions, minlength=6))
    print(f"Most common action: {most_common} ({labels[most_common]})")

def main():
    parser = argparse.ArgumentParser(description="Train DQN for TCP congestion control")
    parser.add_argument('--train_episodes', type=int, default=10, help='Training episodes')
    parser.add_argument('--test_episodes', type=int, default=10, help='Test episodes')
    parser.add_argument('--duration', type=float, default=10.0, help='Sim duration (s)')
    parser.add_argument('--step_time', type=float, default=0.1, help='Time step (s)')
    parser.add_argument('--bottleneck_bandwidth', type=str, default='2Mbps', help='Bottleneck bandwidth')
    parser.add_argument('--bottleneck_delay', type=str, default='0.01ms', help='Bottleneck delay')
    parser.add_argument('--access_bandwidth', type=str, default='10Mbps', help='Access link bandwidth')
    parser.add_argument('--access_delay', type=str, default='2ms', help='Access link delay')
    parser.add_argument('--mtu', type=int, default=400, help='Packet MTU size')
    parser.add_argument('--nLeaf', type=int, default=1, help='Number of leaf nodes')
    parser.add_argument('--port', type=int, default=5555, help='OpenGym port')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load', type=str, default=None, help='Load model weights')
    parser.add_argument('--save', type=str, default='dqn_model.pth', help='Save weights')
    parser.add_argument('--no_train', action='store_true', help='Skip training, only test')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--action_mode', type=str, choices=['default', 'custom'], default='default', help='Action mode')
    parser.add_argument('--action0_scale', type=float, default=5000.0, help='Action 0 cwnd multiplier')
    parser.add_argument('--action1_add', type=int, default=340, help='Action 1 cwnd addition')
    parser.add_argument('--action2_add', type=int, default=1500, help='Action 2 cwnd addition')
    parser.add_argument('--action3_scale', type=float, default=1.2, help='Action 3 cwnd multiplier')
    parser.add_argument('--action4_scale', type=float, default=0.6, help='Action 4 cwnd multiplier')
    parser.add_argument('--action5_scale', type=float, default=0.8, help='Action 5 cwnd multiplier')
    args = parser.parse_args()






    # Set up environment
    sim_args = {
        '--duration': args.duration,
        '--bottleneck_bandwidth': args.bottleneck_bandwidth,
        '--bottleneck_delay': args.bottleneck_delay,
        '--access_bandwidth': args.access_bandwidth,
        '--access_delay': args.access_delay,
        '--mtu': args.mtu,
        '--nLeaf': args.nLeaf,
    }
    env = ns3env.Ns3Env(
        port=args.port,
        stepTime=args.step_time,
        startSim=True,
        simSeed=args.seed,
        simArgs=sim_args,
        debug=False
    )





    # Agent, filter, replay buffer
    tcp_agent = RLTCP(
        action_mode=args.action_mode,
        action0_scale=args.action0_scale,
        action1_add=args.action1_add,
        action2_add=args.action2_add,
        action3_scale=args.action3_scale,
        action4_scale=args.action4_scale,
        action5_scale=args.action5_scale,
    )
    agent = model.DQNAgent()
    filter_obj = state_filter.Filter()   # as defined in your state_filter.py
    replay_buffer = model.ReplayBuffer(capacity=args.buffer_capacity)





    # Load weights if provided
    if args.load and os.path.exists(args.load):
        agent.policy_net.load_state_dict(torch.load(args.load, map_location=device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded weights from {args.load}")





    # ---- Training ----
    if not args.no_train:

        print(f"\n--- Training for {args.train_episodes} episodes ---")
        train_data = []
        for ep in range(args.train_episodes):
            print(f"Episode {ep+1}/{args.train_episodes}", end='')
            data, total_r = run_episode(env, tcp_agent, agent, filter_obj, replay_buffer, train=True, debug=args.debug)
            train_data.append(data)
            print(f"  Steps: {len(data['throughput'])}, Reward: {total_r:.2f}, ε={agent.epsilon:.3f}")






        torch.save(agent.policy_net.state_dict(), args.save)
        print(f"Model saved to {args.save}")
        plot_results(train_data, save_dir='training_plots', title='Training Performance')





        # Reset filter for testing (optional)
        filter_obj = state_filter.Filter()





    # ---- Testing ----
    print(f"\n--- Testing for {args.test_episodes} episodes ---")
    agent.epsilon = 0.09   # exploitation
    test_data = []
    print(agent.epsilon)



    for ep in range(args.test_episodes):
        print(f"Test episode {ep+1}/{args.test_episodes}", end='')
        data, total_r = run_episode(env, tcp_agent, agent, filter_obj, replay_buffer, train=False, debug=args.debug)
        test_data.append(data)
        print(f"  Steps: {len(data['throughput'])}, Reward: {total_r:.2f}")





    plot_results(test_data, save_dir='test_plots', title='Test Performance')
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()