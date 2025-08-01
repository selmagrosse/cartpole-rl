# Cartpole Reinforcement Learning

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 environment in Gymnasium using the Stable Baselines3 library. The model can be trained using hyperparameter optimization with Optuna. 

A detailed explanation of the project and its underlying theory can be read in my [blog post](https://selmagrosse.com/cart-pole-balancing-using-reinforcement-learning/).

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
To train the DQN model, run `main.py` in one of the following modes:
1. Train the DQN model with baseline parameters stored in ```configs/config.yaml```:
```bash
    python main.py --mode train_baseline
```
2. Train the DQN model while running the Optuna for hyperparameter optimization:
```bash
    python main.py --mode train_optuna
```
The updated hyperparameters list is saved in ```configs/optimized_config.yaml```.
3. Test the baseline DQN model:
```bash
    python main.py --mode test_baseline
```
4. Test the optimized DQN model:
```bash
    python main.py --mode test_optuna
```
5. Generate a GIF for the optimized model:
```bash
    python main.py --mode gif
```

## Configuration

The hyperparameters set manually are stored in ```configs/config.yaml``` and the hyperparameters tuned with Optuna are stored in ```configs/optimized_config.yaml```

## Results

After training, the model can be evaluated with Tensorboard. After testing, learning rate (reward values over episodes) is plotted.

## License

This project is licensed under the MIT License.






