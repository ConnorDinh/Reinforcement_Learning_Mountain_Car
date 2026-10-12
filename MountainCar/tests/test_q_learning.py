from src.q_learning import QLearningAgent
import pytest
import numpy as np

def test_q_learning():
    q_learning_agent = QLearningAgent(20, 20, 3, 0.1, 0.99, 0.0)
    assert q_learning_agent.q_table.shape == (20, 20, 3)
    assert np.all(q_learning_agent.q_table == 0)
    state = (3,6)
    q_learning_agent.q_table[3, 6] = [-5.0, -2.0, -8.0]
    action = q_learning_agent.choose_action(state)
    assert action == 1
    q_learning_agent.epsilon = 0.1
    next_state = (4,7)
    action = 2
    reward = -1
    q_learning_agent.q_table[3, 6, 2] = -5.0
    q_learning_agent.q_table[4, 7] = [-6.0, -3.0, -4.0]
    q_learning_agent.update(state, action, reward, next_state, False)
    assert q_learning_agent.q_table[3, 6, 2] == pytest.approx(-4.897)

def test_terminal_update():
    q_learning_agent = QLearningAgent(20, 20, 3, 0.1, 0.99, 0.1)
    state = (3,6)
    next_state = (4,7)
    action = 2
    reward = -1
    q_learning_agent.q_table[3, 6, 2] = -5.0
    q_learning_agent.q_table[4, 7] = [10.0, 20.0, 30.0]
    q_learning_agent.update(state, action, reward, next_state, True)
    assert q_learning_agent.q_table[3, 6, 2] == pytest.approx(-4.6)    
    