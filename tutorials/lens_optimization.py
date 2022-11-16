"""
This is a demo that simulate the evolution of lens
There are some global parameters with following meanings:
len_param_weighting: Weight of focus function
len_param_h: Trapezoidal slices of height
len_param_b: Distance between lens and object
len_param_r: Radius of lens
len_param_eps: Refraction index
len_param_d_init = Init of x
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import imageio

from pypop7.optimizers.es.maes import MAES
from pypop7.optimizers.es.es import ES


len_param_weighting = 0.9
len_param_h = 1
len_param_b = 20
len_param_r = 7
len_param_eps = 1.5
len_param_d_init = 3


def f_lens(x):
    """
    Function that estimate the ability of lens
    """
    n = len(x)
    f_focus = np.sum((len_param_r - ((len_param_h * np.arange(1, n)-0.5) + len_param_b / len_param_h
                                     * (len_param_eps - 1) * np.transpose(np.abs(x[1:n]) - np.abs(x[0:n-1])))) ** 2)
    f_mass = len_param_h * (np.sum(np.abs(x[1:n-2])) + 0.5 * (np.abs(x[0]) + np.abs(x[n-1])))
    qual = len_param_weighting * f_focus + (1 - len_param_weighting) * f_mass
    return qual


def get_path(x):
    """
    Function to get the path of lens according to x
    """
    left_points, right_points = [], []
    temp_height = len_param_r
    for i in range(len(x)):
        if x[i] < 0:
            x[i] *= -1
        left_points.append((-0.5 * x[i], temp_height))
        right_points.append((0.5 * x[i], temp_height))
        temp_height -= 1
    points = left_points
    for i in range(len(right_points)):
        points.append(right_points[-1 * i - 1])
    points.append(left_points[0])
    codes = [Path.MOVETO]
    for i in range(len(points)-2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(points, codes)
    return path


def draw_graph(xs):
    img_path = "./lens"
    file_names = []
    for i in range(len(xs)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = '12'
        ax.set_xlim(-5, 5)
        ax.set_ylim(-8, 8)
        path = get_path(xs[i])
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
        temp_path = img_path + "_" + str(i) + ".png"
        file_names.append(temp_path)
        plt.savefig(temp_path)
    frames = []
    for image_name in file_names:
        im = imageio.imread(image_name)
        frames.append(im)
    imageio.mimsave("./demo_lens.gif", frames, 'GIF', duration=0.3)


class MAES_tutorial(MAES):
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
        results = self._collect_results(fitness, mean)
        results['s'] = s
        results['xs'] = xs
        return results


if __name__ == '__main__':
    d = 15
    problem = {'fitness_function': f_lens,
               'ndim_problem': d,
               'lower_boundary': -2 * np.ones((d,)),
               'upper_boundary': 2 * np.ones((d,))}
    options = {'fitness_threshold': 1e-20,
               'max_function_evaluations': 8e3,
               'seed_rng': 2022,  # not given in the original paper
               'x': len_param_d_init * np.ones((d,)),
               'sigma': 0.3,
               'stagnation': np.Inf,
               'verbose': 50,
               'saving_fitness': 50,
               'is_restart': False}
    solver = MAES_tutorial(problem, options)
    results = solver.optimize()
    print(results)
    draw_graph(results['xs'])
