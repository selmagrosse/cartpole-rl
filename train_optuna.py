import numpy as np
import random
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import yaml
import optuna

# Load hyperparameters from the YAML file
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    cartpole_config = config['CartPole-v1']

SEED = 42
# Set fixed seed for reproducibility
def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

# Define the objective function
def objective(trial):

    # Create environment
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)

    # Define hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    buffer_size = trial.suggest_int("buffer_size", 50000, 200000, step=50000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    target_update_interval = trial.suggest_int("target_update_interval", 100, 10000, step=100)
    train_freq = trial.suggest_int("train_freq", 1, 256, step=4)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.2)

    # Create the DQN model
    model = DQN(
        cartpole_config['policy'],
        env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=cartpole_config['learning_starts'],
        gamma=gamma,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=cartpole_config['gradient_steps'],
        exploration_fraction=exploration_fraction,
        exploration_final_eps=cartpole_config['exploration_final_eps'],
        verbose=1,
        policy_kwargs=eval(cartpole_config['policy_kwargs']),
        tensorboard_log="./dqn_logs/optuna/"
        )

    # Train the the model and display the progress bar
    model.learn(total_timesteps=cartpole_config['n_timesteps'], progress_bar=True)

    # Evaluate the model
    episode_rewards, _ = evaluate_policy(model, env, return_episode_rewards=True)
    mean_reward = np.mean(episode_rewards)

    # Save the trained model
    model.save("dqn_cartpole")

    # Close the environment
    env.close()

    return mean_reward

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=1800)

if len(study.trials) > 0 and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
    optimized_params = study.best_params
    print(f"Best hyperparameters: {optimized_params}")

    # Save the optimized hyperparameters into a new config file
    with open("configs/optimized_config.yaml", "w") as f:
        yaml.dump({"CartPole-v1": optimized_params}, f)
else:
    print("No completed trials found.")