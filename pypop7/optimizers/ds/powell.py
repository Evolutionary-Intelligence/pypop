import numpy as np

from pypop7.optimizers.ds.ds import DS


class POWELL(DS):
    """Powell's Method(POWELL).
    Reference
    ---------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/files/chapter-7.pdf
    (See Algorithm 7.3 (Page 102) for details.)
    M. J. D. Powell
    An Efficient Method for Finding the Minimum of a Function
    of Several Variables Without Calculating Derivative
    Comput.J. 7(2)155-162(1964)
    https://academic.oup.com/comjnl/article-abstract/7/2/155/335330?redirectedFrom=fulltext&login=false
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.q = 0.1

    def initialize(self, args=None, is_restart=False):
        x = self._initialize_x(is_restart)  # initial point
        y = self._evaluate_fitness(x, args)  # fitness
        u = np.zeros((self.ndim_problem, self.ndim_problem))
        for i in range(self.ndim_problem):
            u[i][i] = 1.0
        return x, y, u

    def line_search(self, x=None, y=None, d=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        for sgn in [-1, 1]:
            while True:
                if self._check_terminations():
                    return xx, yy
                x_new = xx + sgn * self.q * d
                y_new = self._evaluate_fitness(x_new)
                if y_new < yy:
                    xx = x_new
                    yy = y_new
                else:
                    break
        return xx, yy

    def iterate(self, args=None, x=None, y=None, u=None):
        xx, yy = np.copy(x), np.copy(y)
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return xx, yy, u
            d = u[i]
            xx, yy = self.line_search(xx, yy, d)
            if i != self.ndim_problem - 1:
                u[i] = u[i+1]
        d = xx - x
        d /= np.linalg.norm(d)
        u[-1] = d
        xx, yy = self.line_search(xx, yy, d)
        if y - yy < 0.05 * self.fitness_threshold:
            self.q *= 0.1
        else:
            self.q = 0.4 * np.sqrt(y - yy)
        return xx, yy, u

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x, y, u = self.initialize(args)
        fitness.append(y)
        while True:
            x, y, u = self.iterate(args, x, y, u)
            if self.record_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
