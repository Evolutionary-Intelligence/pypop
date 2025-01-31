from pypop7.benchmarks.utils import generate_xyz
import numpy as np
import matplotlib.pyplot as plt  # for static plotting

from pypop7.optimizers.es.es import ES  # abstract class for all evolution Strategies
from pypop7.optimizers.es.maes import MAES  # Matrix Adaptation Evolution Strategy


def cd(x):  # from https://arxiv.org/pdf/1610.00040v1.pdf
    return 7.0 * (x[0] ** 2) + 6.0 * x[0] * x[1] + 8.0 * (x[1] ** 2)


# helper function for 2D-plotting
def plot_contour(func, x, y, levels=None, num=200):
    """Plot a 2D contour of the fitness landscape.

    Parameters
    ----------
    func    : func
              benchmarking function.
    x       : list
              x-axis range.
    y       : list
              y-axis range.
    levels  : int
              number of contour lines.
    num     : int
              number of samples in each of x- and y-axis range.
    is_save : bool
              whether or not to save the generated figure in the *local* folder.

    Returns
    -------
    An online figure.
    """
    x, y, z = generate_xyz(func, x, y, num)
    if levels is None:
        plt.contourf(x, y, z, cmap='bone')
        plt.contour(x, y, z)
    else:
        plt.contourf(x, y, z, levels, cmap='cool')
        c = plt.contour(x, y, z, levels, colors='k')
        plt.clabel(c, inline=True, fontsize=12, colors='white')
    plt.xlabel('x')
    plt.ylabel('y')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10


def plot(xs, means):
    for i in range(len(xs)):
        sub_figure = '_' + str(i) + '.png'

        # set title
        plt.title("cd")

        # set boundary
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # draw contour and points
        plot_contour(cd, [-10.0, 10.0], [-10.0, 10.0])
        plt.scatter(xs[i][:, 0], xs[i][:, 1], color='green')
        plt.scatter(means[i][0], means[i][1], color='m')

        # save and show figure
        plt.savefig(sub_figure, dpi=300)
        plt.show()


# <3> - Extend Optimizer Class MAES to Generate Data for Plotting
class MAESPLOT(MAES):  # to overwrite original MAES algorithm for plotting
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        print(z)
        # print(z+mean)
        xs = []  # for plotting
        means = []

        while not self._check_terminations():
            z, d, y = self.iterate(z, d, mean, tm, y, args)
            # print(z+mean)
            if self.saving_fitness and (not self._n_generations % self.saving_fitness):
                xs.append(self.sigma*d+mean)  # for plotting
                means.append(mean.copy())
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            if self.is_restart:
                z, d, mean, s, tm, y = self.restart_reinitialize(z, d, mean, s, tm, y)
        res = self._collect(fitness, y, mean)
        res['xs'] = xs  # for plotting
        return xs, means


if __name__ == '__main__':
    ndim_problem = 2  # dimension of objective function
    problem = {'fitness_function': cd,  # objective (fitness) function
               'ndim_problem': ndim_problem,  # number of dimensionality of objective function
               'lower_boundary': -10.0*np.ones((ndim_problem,)),  # lower boundary of search range
               'upper_boundary': 10.0*np.ones((ndim_problem,))}  # upper boundary of search range
    options = {'max_function_evaluations': 3e3,  # maximum of function evaluations
               'n_individuals': 200,
               'seed_rng': 2022,  # seed of random number generation (for repeatability)
               'x': (7.0, -7.0),
               'sigma': 0.05,  # global step-size of Gaussian search distribution (not necessarily an optimal value)
               'saving_fitness': 4,  # to record best-so-far fitness every 50 function evaluations
               'is_restart': False}  # whether or not to run the (default) restart process
    xs, means = MAESPLOT(problem, options).optimize()
    plot(xs, means)
