import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from pypop7.benchmarks.utils import generate_xyz
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.maes import MAES


def plot_contour(func, x, y, levels=None):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, levels, cmap='bone')
    plt.xlabel('x')
    plt.ylabel('y')


def shi_cd(x):  # fitness function from https://arxiv.org/abs/1610.00040
    return 7 * np.power(x[0], 2) + 6 * x[0] * x[1] + 8 * np.power(x[1], 2)


class MAES1(MAES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        x_data, mean_data = [], []
        x = np.empty((self.n_individuals, self.ndim_problem))
        while True:
            z, d, y = self.iterate(z, d, mean, tm, y, args)  # sample and evaluate offspring population
            for k in range(self.n_individuals):
                x[k] = mean + self.sigma * d[k]
            x_data.append(np.copy(x))
            mean_data.append(np.copy(mean))
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean)
        results['x_data'] = x_data
        results['mean_data'] = mean_data
        return results


if __name__ == '__main__':
    ndim_problem = 2
    pro = {'fitness_function': shi_cd,
           'ndim_problem': ndim_problem}
    opt = {'fitness_threshold': 1e-10,
           'seed_rng': 0,
           'x': np.array([7., -8.]),  # mean
           'sigma': 0.1,
           'verbose_frequency': 5,
           'n_individuals': 250}
    solver = MAES1(pro, opt)
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
        plt.scatter(res['x_data'][i][:, 0], res['x_data'][i][:, 1], c='limegreen', s=3)
        plt.scatter(res['mean_data'][i][0], res['mean_data'][i][1], c='magenta', s=12)
        camera.snap()
    animation = camera.animate()
    animation.save('demo_maes.gif')
