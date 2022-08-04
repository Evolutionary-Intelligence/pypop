import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from pypop7.benchmarks.utils import generate_xyz
from pypop7.optimizers.ds.ds import DS
from pypop7.optimizers.ds.hj import HJ


def plot_contour(func, x, y, levels=None):
    x, y, z = generate_xyz(func, x, y, 500)
    plt.contourf(x, y, z, levels, cmap='bone')
    plt.xlabel('x')
    plt.ylabel('y')


def shi_cd(x):  # fitness function from https://arxiv.org/abs/1610.00040
    return 7 * np.power(x[0], 2) + 6 * x[0] * x[1] + 8 * np.power(x[1], 2)


class HJ1(HJ):
    def iterate(self, args=None, x=None, fitness=None):
        improved, best_so_far_x, best_so_far_y = False, self.best_so_far_x, self.best_so_far_y
        x_data, x_i = np.empty(shape=(4, 2)), 0
        for ii in range(self.ndim_problem):
            for sgn in [-1, 1]:
                if self._check_terminations():
                    return x_data, best_so_far_x
                xx = np.copy(best_so_far_x)
                xx[ii] += sgn * self.sigma
                x_data[x_i] = xx
                x_i += 1
                y = self._evaluate_fitness(xx, args)
                if self.record_fitness:
                    fitness.append(y)
                if y < best_so_far_y:
                    best_so_far_y, improved = y, True
        if not improved:
            self.sigma *= self.gamma  # alpha
        return x_data, best_so_far_x

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x_data, mean_data = [], []
        x, y = self.initialize(args)
        fitness.append(y)
        while True:
            xx, best_so_far_x = self.iterate(args, x, fitness)
            x_data.append(xx)
            mean_data.append(best_so_far_x)
            if self._check_terminations():
                break
            self._n_generations += 1
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
    solver = HJ1(pro, opt)
    res = solver.optimize()
    print(res)
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
    animation.save('demo_hj.gif')
