import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Discrete, Box


class PortfolioEnv(Env):
    def __init__(self, data, initial_balance=100000):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0

        self.action_space = Discrete(len(self.data.columns))
        self.observation_space = Box(low=0, high=np.inf, shape=(len(self.data.columns),))

        self.balance = self.initial_balance
        self.portfolio = [0] * len(self.data.columns)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = [0] * len(self.data.columns)
        return self.data.iloc[self.current_step].values

    def step(self, action):
        current_prices = self.data.iloc[self.current_step].values
        reward = sum(self.portfolio) + self.balance  # Simplified reward
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self.data.iloc[self.current_step].values if not done else None
        return obs, reward, done, {}

def train_model(data):
    env = DummyVecEnv([lambda: PortfolioEnv(data)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_portfolio")
    return model

def optimize_portfolio(data):
    model = PPO.load("ppo_portfolio")
    env = PortfolioEnv(data)
    obs = env.reset()
    portfolio = []

    for _ in range(len(data)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        portfolio.append(action)
        if done:
            break

    return portfolio
