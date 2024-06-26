"""Only for the testing purpose. Online documentation:
    https://pypop.readthedocs.io/en/latest/utils.html
"""
import unittest
import copy
import time

import numpy as np  # engine for numerical computing
import matplotlib.pyplot as plt
from pypop7.benchmarks import base_functions as bf
from pypop7.benchmarks import rotated_functions as rf
from pypop7.benchmarks.utils import generate_xyz
from pypop7.benchmarks.utils import plot_contour
from pypop7.benchmarks.utils import plot_surface
from pypop7.benchmarks.utils import save_optimization
from pypop7.benchmarks.utils import check_optimization
from pypop7.benchmarks.utils import plot_convergence_curve
from pypop7.benchmarks.utils import cholesky_update


# test function for plotting
def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7.0 * (x[0] ** 2) + 6.0 * x[0] * x[1] + 8.0 * (x[1] ** 2)


class TestUtils(unittest.TestCase):
    def test_generate_xyz(self):
        x, y, z = generate_xyz(bf.sphere, [0.0, 1.0], [0.0, 1.0], num=2)
        self.assertTrue(np.allclose(x, np.array([[0.0, 1.0], [0.0, 1.0]])))
        self.assertTrue(np.allclose(y, np.array([[0.0, 0.0], [1.0, 1.0]])))
        self.assertTrue(np.allclose(z, np.array([[0.0, 1.0], [1.0, 2.0]])))

    def test_plot_contour(self):
        # plot smoothness and convexity
        plot_contour(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
        plot_contour(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
        # plot multi-modality
        plot_contour(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-convexity
        plot_contour(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-separability
        plot_contour(cd, [-10.0, 10.0], [-10.0, 10.0])
        # plot ill-condition and non-separability
        rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
        plot_contour(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)

    def test_plot_surface(self):
        # plot smoothness and convexity
        plot_surface(bf.sphere, [-10.0, 10.0], [-10.0, 10.0])
        plot_surface(bf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0])
        # plot multi-modality
        plot_surface(bf.rastrigin, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-convexity
        plot_surface(bf.ackley, [-10.0, 10.0], [-10.0, 10.0])
        # plot non-separability
        plot_surface(cd, [-10.0, 10.0], [-10.0, 10.0])
        # plot ill-condition and non-separability
        rf.generate_rotation_matrix(rf.ellipsoid, 2, 72)
        plot_surface(rf.ellipsoid, [-10.0, 10.0], [-10.0, 10.0], 7)

    def test_save_optimization(self):
        from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
        from pypop7.optimizers.es.res import RES

        dim = 2
        problem = {'fitness_function': rosenbrock,  # to define problem arguments
                   'ndim_problem': dim,
                   'lower_boundary': -5.0 * np.ones((dim,)),
                   'upper_boundary': 5.0 * np.ones((dim,))}
        options = {'max_function_evaluations': 5000,  # to set optimizer options
                   'seed_rng': 2022,
                   'mean': 3.0 * np.ones((dim,)),
                   'sigma': 3.0}  # global step-size may need to be tuned for optimality
        res = RES(problem, options)  # to initialize the black-box optimizer class
        results = res.optimize()  # to run its optimization/evolution process
        save_optimization(results, RES.__name__, rosenbrock.__name__, dim, 1)

    def test_check_optimization(self):
        problem = {'lower_boundary': [-5.0, -7.0],
                   'upper_boundary': [5.0, 7.0]}
        options = {'max_function_evaluations': 7777777}
        results = {'n_function_evaluations': 7777777,
                   'best_so_far_x': np.zeros((2,))}
        check_optimization(problem, options, results)

    def test_plot_convergence_curve(self):
        from pypop7.benchmarks.base_functions import ellipsoid  # function to be minimized
        from pypop7.optimizers.pso.spso import SPSO
        problem = {'fitness_function': ellipsoid,  # define problem arguments
                   'ndim_problem': 2,
                   'lower_boundary': -5.0 * np.ones((2,)),
                   'upper_boundary': 5.0 * np.ones((2,))}
        options = {'max_function_evaluations': 3500,  # set optimizer options
                   'saving_fitness': 1,
                   'seed_rng': 2022}
        spso = SPSO(problem, options)  # initialize the black-box optimizer class
        res = spso.optimize()  # run the optimization process
        plot_convergence_curve(SPSO.__name__, ellipsoid.__name__, 2, results=res)

    def test_plot_convergence_curves(self):
        from pypop7.benchmarks.base_functions import ellipsoid  # function to be minimized
        from pypop7.optimizers.rs.prs import PRS
        from pypop7.optimizers.pso.spso import SPSO
        from pypop7.optimizers.de.cde import CDE
        from pypop7.optimizers.eda.umda import UMDA
        from pypop7.optimizers.es.cmaes import CMAES
        from pypop7.benchmarks.utils import plot_convergence_curves
        algos = [PRS, SPSO, CDE, UMDA, CMAES]
        problem = {'fitness_function': ellipsoid,  # to define problem arguments
                   'ndim_problem': 2,
                   'lower_boundary': -5.0 * np.ones((2,)),
                   'upper_boundary': 5.0 * np.ones((2,))}
        options = {'max_function_evaluations': 5000,  # to set optimizer options
                   'saving_fitness': 1,
                   'sigma': 3.0,
                   'seed_rng': 2022}
        res = []
        for Optimizer in algos:
            optimizer = Optimizer(problem, options)  # to initialize the black-box optimizer class
            res.append(optimizer.optimize())  # to run the optimization process
        plot_convergence_curves(algos, ellipsoid.__name__, 2, results=res)

    def test_cholesky_update(self):
        def cholesky_update_1(rm, z, downdate):  # without Numba
            # https://github.com/scipy/scipy/blob/d20f92fce9f1956bfa65365feeec39621a071932/
            #     scipy/linalg/_decomp_cholesky_update.py
            rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
            alpha[-1], beta[-1] = 1.0, 1.0
            sign = -1.0 if downdate else 1.0
            for r in range(len(z)):
                a = z[r] / rm[r, r]
                alpha[r] = alpha[r - 1] + sign * np.power(a, 2)
                beta[r] = np.sqrt(alpha[r])
                z[r + 1:] -= a * rm[r, r + 1:]
                rm[r, r:] *= beta[r] / beta[r - 1]
                rm[r, r + 1:] += sign * a / (beta[r] * beta[r - 1]) * z[r + 1:]
            return rm.T

        rng = np.random.default_rng(2022)
        ndim = 2 # 2000
        rm_, z_, downdate_ = 2 + rng.random((ndim, ndim)), rng.random(ndim, ), False

        runtime_1 = []
        for i in range(3000):
            rr, zz, dd = copy.deepcopy(rm_), copy.deepcopy(z_), copy.deepcopy(downdate_)
            start_time = time.time()
            cholesky_update_1(rr, zz, dd)
            runtime_1.append(time.time() - start_time)

        runtime_2 = []
        for i in range(3000):
            rr, zz, dd = copy.deepcopy(rm_), copy.deepcopy(z_), copy.deepcopy(downdate_)
            start_time = time.time()
            cholesky_update(rr, zz, dd)
            runtime_2.append(time.time() - start_time)

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = '12'
        plt.figure(figsize=(7, 7))
        plt.grid(True)
        plt.plot(np.cumsum(runtime_1), color='r', label='without Numba', linewidth=2)
        plt.plot(np.cumsum(runtime_2), color='g', label='with Numba', linewidth=2)
        plt.title("Runtime Comparisons on 2000 Dimension",
                  fontsize=24, fontweight='bold')
        plt.xlabel('Number of Iterations', fontsize=20, fontweight='bold')
        plt.ylabel('Runtime (Seconds)', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        plt.legend(fontsize=15, loc='best')
        plt.show()


if __name__ == '__main__':
    unittest.main()
