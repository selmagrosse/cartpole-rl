from train import train_model
from train_optuna import train_model_optuna
from test import test_model
from test_generate_gif import test_model_gif

config_file = "configs/config.yaml"
model_file = "dqn_cartpole_fixed.zip"
model_file_optuna = "dqn_cartpole_optuna.zip"

train_model(config_file=config_file, model_file=model_file, progress_bar=True)
# train_model_optuna(config_file=config_file, model_file=model_file_optuna)

results = test_model(model_file, verbose=True)
print(results)
# results = test_model(model_file_optuna, verbose=True)
# print(results)
# test_model_gif(model_file_optuna, gif_path="cartpole_test.gif")