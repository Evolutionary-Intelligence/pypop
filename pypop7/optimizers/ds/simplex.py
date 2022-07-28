import numpy as np

from pypop7.optimizers.dsm.dsm import DSM


class SIMPLEX(DSM):
    """SIMPLEX
        Reference
        ------------------
        J. A. Nelder, R. Mead
        A simplex method for function minimization
        Comput. J. 7(4): 308-313 (1965)
    """
    def __init__(self, problem, options):
        DSM.__init__(self, problem, options)
        self.gamma = options.get('gamma')
        self.x = None
        self.n_individuals = self.ndim_problem + 1
        if self.n_individuals is None:  # number of offspring, offspring population size (λ: lambda)
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # for small population setting

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals, ))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x(is_restart)
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def iterate(self, x, y):
        order = np.argsort(y)
        l, h = order[0], order[-1]
        p_l, p_h = x[order[0]], x[order[-1]]
        y_l, y_h = y[order[0]], y[order[-1]]
        p_mean = np.zeros((self.ndim_problem, ))
        for i in range(self.n_individuals - 1):
            p_mean += x[order[i]]
        p_mean /= (self.n_individuals - 1)
        p_star = (1 + self.alpha) * p_mean - self.alpha * p_h
        y_star = self._evaluate_fitness(p_star)
        if y_star < y_l:
            # p_star_star = (1 + self.gamma) * p_star - self.gamma * p_mean # graph
            p_star_star = self.gamma * p_star + (1 - self.gamma) * p_mean  # theory
            y_star_star = self._evaluate_fitness(p_star_star)
            if y_star_star < y_l:
                x[h] = p_star_star
            else:
                x[h] = p_star
            y[h] = self._evaluate_fitness(x[h])
        else:
            judge = True
            for i in range(self.n_individuals):
                if i != h and y_star <= y[i]:
                    judge = False
                    break
            if judge is True:
                if y_star <= y_h:
                    x[h] = p_star
                    y[h] = self._evaluate_fitness(x[h])
                p_star_star = self.beta * x[h] + (1 - self.beta) * p_mean
                y_star_star = self._evaluate_fitness(p_star_star)
                if y_star_star > y_h:
                    for i in range(self.n_individuals):
                        x[i] = (x[i] + p_l) / 2
                        y[i] = self._evaluate_fitness(x[i])
                else:
                    x[h] = p_star_star
                    y[h] = self._evaluate_fitness(x[h])
            else:
                x[h] = p_star
                y[h] = self._evaluate_fitness(x[h])
        return x, y

    def optimize(self, fitness_function=None):
        fitness = DSM.optimize(self, fitness_function)
        x, y = self.initialize()
        while True:
            x, y = self.iterate(x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
