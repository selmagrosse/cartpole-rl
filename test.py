import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

# Create environment
env = gym.make("CartPole-v1")

# Load the DQN trained model
model = DQN.load("dqn_cartpole")

# Number of test episodes
num_episodes = 100
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

# Statistics
avg_reward = np.mean(total_rewards)
std_dev = np.std(total_rewards)

print(f"Tested {num_episodes} episodes")
print(f"Average reward: {avg_reward:.2f}")
print(f"Standard deviation: {std_dev:.2f}")