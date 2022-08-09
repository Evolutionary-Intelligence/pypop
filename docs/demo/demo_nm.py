import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from pypop7.benchmarks.utils import generate_xyz
from pypop7.optimizers.ds.ds import DS
from pypop7.optimizers.ds.nm import NM


def plot_contour(func, x, y, levels=None):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, levels, cmap='plasma')
    plt.xlabel('x')
    plt.ylabel('y')


def shi_cd(x):  # fitness function from https://arxiv.org/abs/1610.00040
    return 7 * np.power(x[0], 2) + 6 * x[0] * x[1] + 8 * np.power(x[1], 2)


class NM1(NM):
    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y = self.initialize(args)
        x_data, mean_data = [x], [self.best_so_far_x]
        fitness.extend(y)
        while True:
            x, y = self.iterate(x, y, args, fitness)
            x_data.append(np.copy(x))
            mean_data.append(np.copy(self.best_so_far_x))
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, y = self.restart_initialize(args, x, y, fitness)
        results = self._collect_results(fitness)
        results['x_data'] = x_data
        results['mean_data'] = mean_data
        return results


if __name__ == '__main__':
    ndim_problem = 2
    pro = {'fitness_function': shi_cd,
           'ndim_problem': ndim_problem}
    opt = {'fitness_threshold': 1e-10,
           'seed_rng': 0,
           'x': np.array([7., -8.]),
           'sigma': 0.9,
           'record_fitness': True,
           'record_fitness_frequency': 1}
    solver = NM1(pro, opt)
    res = solver.optimize()
    fig = plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'
    camera = Camera(fig)
    for i in range(len(res['mean_data'])):
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plt.yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plot_contour(shi_cd, [-10, 10], [-10, 10], [0, 10, 100, 500, 1000, 2000])
        plt.scatter(res['x_data'][i][:, 0], res['x_data'][i][:, 1], c='white', s=3)
        plt.scatter(res['mean_data'][i][0], res['mean_data'][i][1], c='magenta', s=12)
        camera.snap()
    animation = camera.animate()
    animation.save('demo_nm.gif')
