import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from pypop7.benchmarks.utils import generate_xyz
from pypop7.optimizers.core.optimizer import Optimizer
from pypop7.optimizers.rs.prs import PRS


def plot_contour(func, x, y, levels=None):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, levels, cmap='bone')
    plt.xlabel('x')
    plt.ylabel('y')


def shi_cd(x):  # fitness function from https://arxiv.org/abs/1610.00040
    return 7 * np.power(x[0], 2) + 6 * x[0] * x[1] + 8 * np.power(x[1], 2)


class PRS1(PRS):
    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x_data = []
        is_initialization = True
        while True:
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()
            if not self._n_generations % self.verbose_frequency or self._check_terminations():
                x_data.append(x)
            y = self._evaluate_fitness(x, args)
            if self.record_fitness:
                fitness.append(y)
            self._print_verbose_info(y)
            if self._check_terminations():
                break
            self._n_generations += 1
        results = self._collect_results(fitness)
        results['x_data'] = x_data
        return results


if __name__ == '__main__':
    ndim_problem = 2
    pro = {'fitness_function': shi_cd,
           'ndim_problem': ndim_problem,
           'lower_boundary': -10 * np.ones((ndim_problem,)),
           'upper_boundary': 10 * np.ones((ndim_problem,))}
    opt = {'fitness_threshold': 2.5e-3,
           'seed_rng': 0,
           'x': np.array([7., -8.]),
           'record_fitness': False,
           'record_fitness_frequency': 1,
           'verbose_frequency': 1e4}
    solver = PRS1(pro, opt)
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
        plt.scatter(res['x_data'][i][0], res['x_data'][i][1], c='magenta', s=12)
        camera.snap()
    animation = camera.animate()
    animation.save('demo_prs.gif')
