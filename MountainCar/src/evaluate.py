import gymnasium as gym
from src.discretizer import Discretizer
from src.q_learning import QLearningAgent
import numpy as np

def evaluate(config, seed):
    env = gym.make("MountainCar-v0")
    discretizer = Discretizer(config[0], config[1])
    agent = QLearningAgent(config[0], config[1], 3, 0.1, 0.99, 0.0)
    agent.q_table = np.load(f'results/models/q_table_{config[0]}x{config[1]}_seed{seed}.npy')
    # total_q = np.sum(agent.q_table)
    # print(total_q)
    # print(np.sum(agent.q_table == 0))
    # print(np.sum(agent.q_table > 0))
    # print(np.sum(agent.q_table < 0))
    # print(np.min(agent.q_table))
    # print(np.max(agent.q_table))
    # print(total_q/1200)

    evaluation_episodes = 100
    rewards = []
    successes = []
    for evaluation in range(evaluation_episodes):
        state, info = env.reset(seed=1000+evaluation)
        state = discretizer.discretize(state)
        total_reward = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discretizer.discretize(next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        successes.append(terminated)
    print("Average reward: ", sum(rewards)/len(rewards))
    print("Success rate: ", successes.count(True)/len(successes) * 100, "%")
    average_reward = sum(rewards) / len(rewards)
    success_rate = successes.count(True) / len(successes) * 100
    env.close()
    return average_reward, success_rate