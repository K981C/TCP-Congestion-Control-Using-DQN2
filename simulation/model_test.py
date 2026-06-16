import gym
import ns3gym
import MyAgent
from ns3gym import ns3env

#env = gym.make('ns3-v0')  <--- causes some errors with the new OpenAI Gym framework, please use ns3env.Ns3Env()
env = ns3env.Ns3Env()
obs = env.reset()
agent = MyAgent.Agent()

while True:
  action = agent.get_action(obs)
  obs, reward, done, info = env.step(action)

  if done:
    break
env.close()