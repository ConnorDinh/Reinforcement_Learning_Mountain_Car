import gymnasium as gym
from src.discretizer import Discretizer
from src.q_learning import QLearningAgent
import numpy as np
from pathlib import Path
import random

def train(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    discretizer = Discretizer(config[0], config[1])
    agent = QLearningAgent(config[0], config[1], 3, 0.1, 0.99, 1.0)
    episodes = 10000
    reward_history = []
    success_history = []
    for episode in range(episodes):
        state, info = env.reset()
        state = discretizer.discretize(state)
        total_reward = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discretizer.discretize(next_state)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        reward_history.append(total_reward)
        success_history.append(terminated)
        agent.epsilon = max(0.01, agent.epsilon* 0.9995)
        # if (episode+1) % 100 == 0:
        #     recent_reward = reward_history[-100:]
        #     recent_successes = success_history[-100:]
        #     print("Episode: ", episode+1)
        #     print("Average reward: ", sum(recent_reward)/100.0)
        #     print("Success rate: ", recent_successes.count(True), "%")
        #     print("Current epsilon: ", agent.epsilon)
    Path("results/models").mkdir(parents=True, exist_ok=True)
    np.save(f'results/models/q_table_{str(config[0])}x{str(config[1])}_seed{seed}.npy', agent.q_table)
    np.save(f'results/models/reward_history_{str(config[0])}x{str(config[1])}_seed{seed}.npy', reward_history)
    np.save(f'results/models/success_history_{str(config[0])}x{str(config[1])}_seed{seed}.npy', success_history)

    env.close()
