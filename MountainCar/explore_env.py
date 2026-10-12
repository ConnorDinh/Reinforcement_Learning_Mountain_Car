import gymnasium as gym

env = gym.make("MountainCar-v0")

for episode in range(10):
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    is_done = False

    while not is_done:

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        observation = next_state
    
        is_done = terminated or truncated
    print(episode)
    print(total_reward)
    print(steps)
    print(terminated)

env.close()