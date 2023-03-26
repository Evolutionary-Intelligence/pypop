"""This is a simple demo to optimize a linear controller on the popular `gymnasium` platform:
    https://github.com/Farama-Foundation/Gymnasium

    For benchmarking, please use the more challenging MuJoCo tasks: https://mujoco.org/
"""
import numpy as np
import gymnasium as gym  # to be installed from https://github.com/Farama-Foundation/Gymnasium

from pypop7.optimizers.es.maes import MAES as Solver


env = gym.make('CartPole-v1', render_mode='human')
action_dim = 2  # for action probability space


class Controller:  # linear controller for simplicity
    def __init__(self, obs):
        self.observation = obs

    def __call__(self, x):
        rewards = 0
        self.observation, _ = env.reset()
        for i in range(1000):
            action = np.matmul(x.reshape(action_dim, -1), self.observation[:, np.newaxis])
            actions = np.sum(action)
            prob_left, prob_right = action[0]/actions, action[1]/actions  # seen as a probability
            action = 1 if prob_left < prob_right else 0
            self.observation, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            if terminated or truncated:
                return -rewards  # for minimization (rather than maximization) 
        return -rewards  # to negate rewards


if __name__ == '__main__':
    observation, _ = env.reset(seed=2023)
    controller = Controller(observation)
    pro = {'fitness_function': controller,
           'ndim_problem': len(observation)*action_dim,
           'lower_boundary': -10*np.ones((len(observation)*action_dim,)),
           'upper_boundary': 10*np.ones((len(observation)*action_dim,))}
    opt = {'max_function_evaluations': 1e4,
           'seed_rng': 0,
           'sigma': 3.0,
           'verbose': 1}
    solver = Solver(pro, opt)
    print(solver.optimize())
    env.close()
