import argparse
from train_baseline import train_model
from train_optuna import train_model_optuna
from test import test_model
from test_generate_gif import test_model_gif

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train_baseline", "train_optuna", "test_baseline", "test_optuna", "gif"],
                    required=True)
args = parser.parse_args()

config_file = "configs/config.yaml"

if args.mode == "train_baseline":        
    model_file = "dqn_cartpole_baseline.zip"
    train_model(config_file=config_file, model_file=model_file, progress_bar=True)

elif args.mode == "train_optuna":
    model_file_optuna = "dqn_cartpole_optimized.zip"
    train_model_optuna(config_file=config_file, model_file=model_file_optuna)

elif args.mode == "test_baseline":
    model_file = "dqn_cartpole_baseline.zip"
    results = test_model(model_file, verbose=True)
    print(results)

elif args.mode == "test_optuna":
    model_file = "dqn_cartpole_optimized.zip"
    results = test_model(model_file, verbose=True)
    print(results)

elif args.mode == "gif":
    model_file = "dqn_cartpole_optimized.zip"
    test_model_gif(model_file, gif_path="cartpole_optimized.gif")