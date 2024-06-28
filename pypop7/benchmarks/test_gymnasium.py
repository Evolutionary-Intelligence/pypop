import numpy as np

from pypop7.benchmarks.gymnasium import Cartpole
from pypop7.optimizers.es.maes import MAES as Controller


def testCartpole():
    env = Cartpole()
    pro = {'fitness_function': env,
           'ndim_problem': len(env.observation)*env.action_dim,
           'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
           'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
    opt = {'max_function_evaluations': 7,
           'seed_rng': 0,
           'sigma': 3.0,
           'verbose': 1}
    controller = Controller(pro, opt)
    print(controller.optimize())
