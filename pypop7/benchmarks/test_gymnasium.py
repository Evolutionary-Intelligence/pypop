import numpy as np

from pypop7.benchmarks.gymnasium import Cartpole
# from pypop7.benchmarks.gymnasium import Ant
# from pypop7.benchmarks.gymnasium import HalfCheetah
# from pypop7.benchmarks.gymnasium import Hopper
# from pypop7.benchmarks.gymnasium import Humanoid
# from pypop7.benchmarks.gymnasium import Swimmer
# from pypop7.benchmarks.gymnasium import Walker2d
from pypop7.optimizers.es.maes import MAES as Controller


def testCartpole():
    env = Cartpole()
    pro = {'fitness_function': env,
           'ndim_problem': len(env.observation) * env.action_dim,
           'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
           'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
    opt = {'max_function_evaluations': 7,
           'seed_rng': 0,
           'sigma': 3.0,
           'verbose': 1}
    controller = Controller(pro, opt)
    print(controller.optimize())


# def testAnt():
#     env = Ant()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
#
#
# def testHalfCheetah():
#     env = HalfCheetah()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
#
#
# def testHopper():
#     env = Hopper()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
#
#
# def testHumanoid():
#     env = Humanoid()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
#
#
# def testSwimmer():
#     env = Swimmer()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
#
#
# def testWalker2d():
#     env = Walker2d()
#     pro = {'fitness_function': env,
#            'ndim_problem': len(env.observation) * env.action_dim,
#            'lower_boundary': -10 * np.ones((len(env.observation) * env.action_dim,)),
#            'upper_boundary': 10 * np.ones((len(env.observation) * env.action_dim,))}
#     opt = {'max_function_evaluations': 7,
#            'seed_rng': 0,
#            'sigma': 3.0,
#            'verbose': 1}
#     controller = Controller(pro, opt)
#     print(controller.optimize())
