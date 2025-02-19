# Cartpole Reinforcement Learning

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 environment in Gymnasium using the Stable Baselines3 library. The model can be trained using hyperparameter optimization with Optuna.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/selmagrosse/cartpole-rl.git
   cd cartpole-rl
2. Create a virtual environment and run it:
   ```bash
    python -m venv env
    source env/bin/activate
3. Install dependencies:
   ```bash
    pip install -r requirements.txt

## Training and testing
To train the DQN model, run:

```bash
    python main.py
```
This will execute both the training with manually set hyperparameters and the training with Optuna hyperparameters optimization.

## Configuration

The hyperparameters tuned manually are stored in ```configs/config.yaml``` and the hyperparameters tuned with Optuna are stored in ```configs/optimized_config.yaml```

## Results

After training, the model can be evaluated with tensorboard. After testing, learning rate (reward values over episodes) is plotted.

## License

This project is licensed under the MIT License.






