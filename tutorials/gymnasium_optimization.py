"""This is a simple demo that optimizes on the well-designed `gymnasium` platform.
    The code of gymnaisum can be found in:
    https://github.com/Farama-Foundation/Gymnasium
    Use pip install "gymnasium[all], to install all dependencies
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces.discrete import Discrete

from pypop7.optimizers.es.maes import MAES as Solver
env = gym.make("CartPole-v1", render_mode="human")


class Controller:
    def __init__(self, observation):
        self.observation = observation

    def __call__(self, x):
        total_reward = 0
        self.observation, info = env.reset()
        if type(env.action_space) == Discrete:
            dim = 2
        else:
            dim = len(env.action_space)
        for i in range(1000):
            action = np.matmul(x.reshape(dim, -1), self.observation[:, np.newaxis])
            total_action = np.sum(action)
            a, b = action[0] / total_action, action[1] / total_action
            if a < b:
                choose_action = 1
            else:
                choose_action = 0
            self.observation, reward, terminated, truncated, info = env.step(choose_action)
            total_reward += reward
            if terminated or truncated:
                return -total_reward
        return -total_reward


observation, info = env.reset(seed=42)
if type(env.action_space) == Discrete:
    dim = 2
else:
    dim = len(env.action_space)

controller = Controller(observation)
pro = {'fitness_function': controller,
       'ndim_problem': len(observation) * dim,
       'lower_boundary': -10 * np.ones((len(observation) * dim,)),
       'upper_boundary': 10 * np.ones((len(observation) * dim,))}
opt = {'max_function_evaluations': 1e4,
       'seed_rng': 0,
       'sigma': 3.0,
       'verbose': 1,
       'saving_fitness': 0}
solver = Solver(pro, opt)
res = solver.optimize()
env.close()
