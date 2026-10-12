import numpy as np
import random

class QLearningAgent:
    def __init__(self, position_bins, velocity_bins, num_actions, alpha, gamma, epsilon):
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((self.position_bins, self.velocity_bins, self.num_actions))

    def choose_action(self, discrete_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.q_table[discrete_state[0], discrete_state[1], :]
        max_value = np.max(q_values)
        best_actions = np.where(q_values == max_value)[0]
        return int(random.choice(best_actions))

    def update(self, state, action, reward, next_state, terminated):
        old_q = self.q_table[state[0]][state[1]][action]
        if terminated:
            target = reward
        else:
            q_values = self.q_table[next_state[0], next_state[1], :]
            best_next_q = np.max(q_values)
            target = reward + self.gamma*best_next_q
        self.q_table[state[0]][state[1]][action] += self.alpha*(target -  old_q)