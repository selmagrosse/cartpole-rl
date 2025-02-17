import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
import yaml

# Load hyperparameters from the YAML file
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    cartpole_config = config['CartPole-v1']

# Create environment with logging
env = gym.make("CartPole-v1")

# Create the DQN model
model = DQN(
    cartpole_config['policy'],
    env, 
    learning_rate=cartpole_config['learning_rate'],
    buffer_size=cartpole_config['buffer_size'],
    batch_size=cartpole_config['batch_size'],
    learning_starts=cartpole_config['learning_starts'],
    gamma=cartpole_config['gamma'],
    target_update_interval=cartpole_config['target_update_interval'],
    train_freq=cartpole_config['train_freq'],
    gradient_steps=cartpole_config['gradient_steps'],
    exploration_fraction=cartpole_config['exploration_fraction'],
    exploration_final_eps=cartpole_config['exploration_final_eps'],
    verbose=1,
    policy_kwargs=eval(cartpole_config['policy_kwargs']),
    tensorboard_log="./dqn_logs/"
    )

# Train the the model and display the progress bar
model.learn(total_timesteps=cartpole_config['n_timesteps'], progress_bar=True)

# Save the trained model
model.save("dqn_cartpole")

# Close the environment
env.close()
        