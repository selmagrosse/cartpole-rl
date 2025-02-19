from train import train_model
from train_optuna import train_model_optuna
from test import test_model

config_file = "configs/config.yaml"
model_file = "dqn_cartpole.zip"
model_file_optuna = "dqn_cartpole_optuna.zip"
train_model(config_file=config_file, model_file=model_file, progress_bar=True)
train_model_optuna(config_file=config_file, model_file=model_file_optuna)

results = test_model(model_file, verbose=True)
print(results)
results = test_model(model_file_optuna, verbose=True)
print(results)