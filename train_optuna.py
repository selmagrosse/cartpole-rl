import numpy as np
import random
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import torch
import yaml
import optuna

SEED = 42
# Set fixed seed for reproducibility
def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def stop_if_500(study, trial, fixed_config):
    if trial.value and trial.value == 500:
        print("Max reward 500 achieved. Early stopping...")

        full_config = fixed_config.copy()
        full_config.update(trial.params)
        config = {"CartPole-v1": full_config}
        with open("configs/optimized_config.yaml", "w") as f:
            yaml.dump(config, f)

        study.stop()

# Define the objective function
def objective(trial, cartpole_config, model_file="dqn_cartpole_optimized.zip"):

    # Create training environment
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)

    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    buffer_size = trial.suggest_int("buffer_size", 50000, 200000, step=50000)
    learning_starts = trial.suggest_int("learning_starts", 100, 1000, step=100)
    target_update_interval = trial.suggest_int("target_update_interval", 10, 10000, step=1000)
    train_freq = trial.suggest_categorical("train_freq", [4, 16, 32, 64, 128, 256])
    gradient_steps = trial.suggest_categorical("gradient_steps", [4, 16, 32, 64, 128, 256])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.2)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    # Create the DQN model
    model = DQN(
        cartpole_config['policy'],
        env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        gamma=cartpole_config['gamma'],
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=0,
        policy_kwargs=eval(cartpole_config['policy_kwargs']),
        seed=42,
        tensorboard_log="./dqn_logs/optuna/"
        )

    # Train the the model
    model.learn(total_timesteps=cartpole_config['n_timesteps'], progress_bar=False)

    # Evaluate the model
    episode_rewards, _ = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
    mean_reward = np.mean(episode_rewards)

    try:
        best_value = trial.study.best_value
    except ValueError:  # No completed trials exist yet
        best_value = float("-inf")  

    if mean_reward > best_value:
        print(f"New best model found with reward {mean_reward:.2f}, saving...")
        model.save(model_file)

    # Close the environment
    env.close()

    return mean_reward

def train_model_optuna(config_file, model_file="dqn_cartpole_optimized.zip", n_trials=200, timeout=3600):
    
    set_seed(SEED)
    # Load hyperparameters from the YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        cartpole_config = config['CartPole-v1']

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, cartpole_config), 
        n_trials=n_trials, 
        timeout=timeout, 
        callbacks=[lambda study, trial: stop_if_500(study, trial, cartpole_config)])

    if len(study.trials) > 0 and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
        optimized_params = study.best_params
        print(f"Best hyperparameters: {optimized_params}")

        # Merge fixed and optimized parameters
        full_config = cartpole_config.copy()
        full_config.update(optimized_params)

        # Save full config to YAML
        with open("configs/optimized_config.yaml", "w") as f:
            yaml.dump({"CartPole-v1": full_config}, f)
    else:
        print("No completed trials found.")