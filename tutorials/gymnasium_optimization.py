"""This is a simple demo to optimize a linear controller on the popular `gymnasium` platform:
    https://github.com/Farama-Foundation/Gymnasium

    $ pip install gymnasium
    $ pip install gymnasium[classic-control]

    For benchmarking, please use e.g. the more challenging MuJoCo tasks: https://mujoco.org/
"""
import numpy as np
import gymnasium as gym  # to be installed from https://github.com/Farama-Foundation/Gymnasium

from pypop7.optimizers.es.maes import MAES as Solver


class Controller:  # linear controller for simplicity
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.observation, _ = self.env.reset()
        self.action_dim = 2  # for action probability space

    def __call__(self, x):
        rewards = 0
        self.observation, _ = self.env.reset()
        for i in range(1000):
            action = np.matmul(x.reshape(self.action_dim, -1), self.observation[:, np.newaxis])
            actions = np.sum(action)
            prob_left, prob_right = action[0]/actions, action[1]/actions  # seen as a probability
            action = 1 if prob_left < prob_right else 0
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            rewards += reward
            if terminated or truncated:
                return -rewards  # for minimization (rather than maximization) 
        return -rewards  # to negate rewards


if __name__ == '__main__':
    c = Controller()
    pro = {'fitness_function': c,
           'ndim_problem': len(c.observation)*c.action_dim,
           'lower_boundary': -10*np.ones((len(c.observation)*c.action_dim,)),
           'upper_boundary': 10*np.ones((len(c.observation)*c.action_dim,))}
    opt = {'max_function_evaluations': 1e4,
           'seed_rng': 0,
           'sigma': 3.0,
           'verbose': 1}
    solver = Solver(pro, opt)
    print(solver.optimize())
    c.env.close()
