"""This is a simple demo to optimize a linear controller on the popular `gymnasium` platform:
    https://github.com/Farama-Foundation/Gymnasium
"""
import numpy as np
import gymnasium as gym

from pypop7.optimizers.es.maes import MAES as Solver


env = gym.make('CartPole-v1', render_mode='human')
dim = 2  # for action probability space


class Controller:
    def __init__(self, observation):
        self.observation = observation
        self.dim = dim  # for action probability space

    def __call__(self, x):
        rewards = 0
        self.observation, _ = env.reset()
        for i in range(1000):
            action = np.matmul(x.reshape(dim, -1), self.observation[:, np.newaxis])
            actions = np.sum(action)
            prob_left, prob_right = action[0]/actions, action[1]/actions  # seen as a probability
            if prob_left < prob_right:
                action = 1
            else:
                action = 0
            self.observation, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            if terminated or truncated:
                return -rewards  # for minimization (rather than maximization) 
        return -rewards  # to negate rewards


if __name__ == '__main__':
    observation, _ = env.reset(seed=2023)
    controller = Controller(observation)
    pro = {'fitness_function': controller,
           'ndim_problem': len(observation)*dim,
           'lower_boundary': -10*np.ones((len(observation)*dim,)),
           'upper_boundary': 10*np.ones((len(observation)*dim,))}
    opt = {'max_function_evaluations': 1e4,
           'seed_rng': 0,
           'sigma': 3.0,
           'verbose': 1,
           'saving_fitness': 0}
    solver = Solver(pro, opt)
    res = solver.optimize()
    env.close()
