from train_optuna import train_model_optuna
from test import test_model

config_file = "configs/config.yaml"
train_model_optuna(config_file=config_file)

results = test_model("dqn_cartpole_optuna.zip", verbose=True)
print(results)