import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from pypop7.benchmarks.utils import generate_xyz
from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.rhc import RHC


def plot_contour(func, x, y, levels=None):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, levels, cmap='bone')
    plt.xlabel('x')
    plt.ylabel('y')


def shi_cd(x):  # fitness function from https://arxiv.org/abs/1610.00040
    return 7 * np.power(x[0], 2) + 6 * x[0] * x[1] + 8 * np.power(x[1], 2)


class RHC1(RHC):
    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x_data = []
        best_data = []
        is_initialization = True
        while True:
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()
            y = self._evaluate_fitness(x, args)
            best_data.append(self.best_so_far_x)
            x_data.append(x)
            if self.record_fitness:
                fitness.append(y)
            self._print_verbose_info(y)
            if self._check_terminations():
                break
            self._n_generations += 1
        results = self._collect_results(fitness)
        results['x_data'] = x_data
        results['best_data'] = best_data
        return results


if __name__ == '__main__':
    ndim_problem = 2
    pro = {'fitness_function': shi_cd,
           'ndim_problem': ndim_problem,
           'lower_boundary': -10 * np.ones((ndim_problem,)),
           'upper_boundary': 10 * np.ones((ndim_problem,))}
    opt = {'fitness_threshold': 1e-2,
           'seed_rng': 0,
           'x': np.array([7., -8.]),
           'sigma': 1.0,
           'record_fitness': False,
           'record_fitness_frequency': 1,
           'verbose_frequency': 1}
    solver = RHC1(pro, opt)
    res = solver.optimize()
    print(res)
    fig = plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'
    camera = Camera(fig)
    for i in range(len(res['x_data'])):
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plt.yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plot_contour(shi_cd, [-10, 10], [-10, 10], [0, 10, 100, 500, 1000, 2000])
        plt.scatter(res['best_data'][i][0], res['best_data'][i][1], c='magenta', s=12)
        plt.scatter(res['x_data'][i][0], res['x_data'][i][1], c='limegreen', s=3)
        camera.snap()
    animation = camera.animate()
    animation.save('demo_rhc.gif')
