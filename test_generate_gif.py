import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
import imageio

def test_model_gif(model_path, gif_path="cartpole_test.gif"):

    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Load the DQN trained model
    model = DQN.load(model_path)

    done = False
    truncated = False
    obs, _ = env.reset()
    episode_reward = 0
    frames = []

    while not done and not truncated:
        frame = env.render()
        frames.append(frame)
            
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

    env.close()

   # Save to gif
    imageio.mimsave(gif_path, frames, fps=30)
    print("Reward: {episode_reward:.2f}")
