# Implments some Macroeconomic environments on open ai gym

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class GrowthModel(gym.Env):
    metadata = {'render_modes': ['None', 'Graph', 'Desc', 'Verbose']}

    def __init__(self, time_periods=30, k0 = 2, beta=0.9, alpha=0.3, delta=0.2, sigma=1, render_mode=None, random_k0=False):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.time_periods = time_periods
        self.time = 0

        # Parameters
        self.k0 = k0
        self.delta = delta
        self.beta = beta
        self.alpha = alpha
        self.A = A
        self.sigma = sigma
        self.random_k0 = random_k0

        # Functions
        if self.sigma == 1:
            self.utility = lambda c: np.log(max(c, 1e-6))
        else:
            self.utility = lambda c: (c ** (1-self.sigma))/(1-self.sigma)

        self.production = lambda k: self.A * (k ** self.alpha)
        self.available_output = lambda k: self.production(k) + (1-self.delta) * k
        self.consumption = lambda k, kprime: self.available_output(k) - kprime

        # Random Initialization (Takes k0 as maximum possible initialization), taking 20 points for now
        if self.random_k0:
            self.possible_k = np.linspace(0.1, k0, 20)
            self.k0 = np.random.choice(self.possible_k)

        # Spaces
        self.action_space = spaces.Box(low = 0, high = self.available_output(self.k0), shape=(1,), dtype=np.float32)
        # Only choice is capital tomorrow.
        self.observation_space = spaces.Box(low=0, high=self.available_output(self.k0), shape=(1,), dtype=np.float32)
        # Capital Tomorrow is observed


        self.k = self.k0
        self.reward_range = (-np.inf, np.inf)
        self.cumulative_reward = 0
        self.history = pd.DataFrame(columns=['Consumption', 'Capital+1', 'Capital', 'Output'])

    def reset(self, seed = None, return_info = False, options = None):
        reward_achieved = self.cumulative_reward
        self.cumulative_reward = 0
        self.time = 0

        if self.random_k0:
            self.k0 = np.random.choice(self.possible_k)

        history_achieved = self.history.copy()
        self.history =  pd.DataFrame(columns=['Consumption', 'Capital+1', 'Capital', 'Output'])

        self.action_space = spaces.Box(low=0, high = self.available_output(self.k0), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high = self.available_output(self.k0), shape=(1,), dtype=np.float32)

        info = {'Reward': reward_achieved, 'History': history_achieved}
        obs = np.array([self.k0], dtype=np.float32)
        self.k = self.k0

    #TODO: RENDER MODES
        if return_info:
            return obs, info

        return obs

    def step(self, action):
        kprime = action[0]
        consumption = self.consumption(self.k, kprime)
        output = self.production(self.k)
        goods_next_period = self.available_output(kprime)

        self.history.loc[self.time] = [consumption, self.k, kprime, output]
        self.k = kprime

        self.action_space = spaces.Box(low=0, high = goods_next_period, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high = goods_next_period, shape=(1,), dtype=np.float32)

        reward = self.utility(consumption)
        self.cumulative_reward += (self.beta ** self.time) * reward
        obs = np.array([kprime])


        self.time += 1
        done = self.time == self.time_periods








class RBCModel(gym.Env):
    pass

class AiyagariModel(gym.Env):
    pass