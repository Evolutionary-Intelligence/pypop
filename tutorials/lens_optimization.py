"""This is a simple demo that optimizes the lens shape.

    Reference
    ---------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import imageio.v2 as imageio

from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.maes import MAES


# set parameters for lens shape optimization
weight = 0.9  # weight of focus function
r = 7  # radius of lens
h = 1  # trapezoidal slices of height
b = 20  # distance between lens and object
eps = 1.5  # refraction index
d_init = 3  # initialization


# define objective function to be minimized
def f_lens(x):
    n = len(x)
    focus = r - ((h*np.arange(1, n) - 0.5) + b/h*(eps - 1)*np.transpose(np.abs(x[1:]) - np.abs(x[:(n-1)])))
    mass = h*(np.sum(np.abs(x[1:(n-1)])) + 0.5*(np.abs(x[0]) + np.abs(x[n-1])))
    return weight*np.sum(focus**2) + (1.0 - weight)*mass


def get_path(x):
    left, right, height = [], [], r
    for i in range(len(x)):
        if x[i] < 0:
            x[i] *= -1
        left.append((-0.5*x[i], height))
        right.append((0.5*x[i], height))
        height -= 1
    points = left
    for i in range(len(right)):
        points.append(right[-i - 1])
    points.append(left[0])
    codes = [Path.MOVETO]
    for i in range(len(points) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    return Path(points, codes)


def plot(xs):
    file_names, frames = [], []
    for i in range(len(xs)):
        sub_figure = "_" + str(i) + ".png"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = '12'
        ax.set_xlim(-10, 10)
        ax.set_ylim(-8, 8)
        path = get_path(xs[i])
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
        plt.savefig(sub_figure)
        file_names.append(sub_figure)
    for image in file_names:
        frames.append(imageio.imread(image))
    imageio.mimsave("./lens_optimization.gif", frames, 'GIF', duration=0.3)


class MAES1(MAES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        xs = [mean.copy()]
        while True:
            # sample and evaluate offspring population
            z, d, y = self.iterate(z, d, mean, tm, y, args)
            if self.saving_fitness and (not self._n_generations % self.saving_fitness):
                xs.append(self.best_so_far_x)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._print_verbose_info(y)
            self._n_generations += 1
            if self.is_restart:
                z, d, mean, s, tm, y = self.restart_reinitialize(z, d, mean, s, tm, y)
        res = self._collect_results(fitness, mean)
        res['s'] = s
        res['xs'] = xs
        return res


if __name__ == '__main__':
    dim = 15  # dimension of objective function
    problem = {'fitness_function': f_lens,
               'ndim_problem': dim,
               'lower_boundary': -5*np.ones((dim,)),
               'upper_boundary': 5*np.ones((dim,))}
    options = {'max_function_evaluations': 7e3,
               'seed_rng': 2022,
               'x': d_init*np.ones((dim,)),
               'sigma': 0.3,
               'saving_fitness': 50,
               'is_restart': False}
    results = MAES1(problem, options).optimize()
    plot(results['xs'])
