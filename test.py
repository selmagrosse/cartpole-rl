import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window_size):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# Create environment
env = gym.make("CartPole-v1")

# Load the DQN trained model
model = DQN.load("dqn_cartpole")

# Number of test episodes
num_episodes = 500
# Track performance with total rewards
total_rewards = []

# Run the agent
for episode in range(num_episodes):
    done = False
    truncated = False
    obs, _ = env.reset()
    episode_reward = 0
    while not done and not truncated:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        if done or truncated:
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break
                  
env.close()

smoothed_rewards = moving_average(total_rewards, window_size=10)

# Statistics
avg_reward = np.mean(total_rewards)
std_dev = np.std(total_rewards)

print(f"Tested {num_episodes} episodes")
print(f"Average reward: {avg_reward:.2f}")
print(f"Standard deviation: {std_dev:.2f}")

# Plot rewards
plt.figure(figsize=(10,5))
plt.plot(range(1, num_episodes + 1), total_rewards, label="Raw rewards")
plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed rewards", linewidth=2)
plt.axhline(avg_reward, label=f"Average reward = {avg_reward:.2f}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Testing performance of DQN on CartPole environment")
plt.legend()
plt.grid()
plt.show()
