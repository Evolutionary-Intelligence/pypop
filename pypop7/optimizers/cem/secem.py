import numpy as np

from pypop7.optimizers.cem.cem import CEM
import colorednoise

class SECEM(CEM):
    """Sample-efficient Cross-Entropy Method
    Reference
    ---------------
    C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stuckler, M. Rolinek, G. Martius
    Sample-efficient Cross-Entropy Method for Real-time Planning
    https://arxiv.org/pdf/2008.06389.pdf
    """
    def __init__(self, problem, options):
        CEM.__init__(self, problem, options)
        self.alpha = 0.1
        self.fraction_reused_elites = options.get('fraction_reused_elites')
        self.decay = 1.25  # decay parameter used in N
        self.noise_beta = options.get('noise_beta')

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)
        y = np.empty((self.n_individuals,))
        x = np.empty((self.n_individuals, self.ndim_problem))
        elite_x = np.empty((self.n_parents, self.ndim_problem))
        elite_y = np.empty((self.n_parents,))
        return mean, x, elite_x, y, elite_y

    def re_initialize(self):
        y = np.empty((self.n_individuals,))
        return y

    def iterate(self, mean, x, y):
        for i in range(self.n_individuals):
            x[i] = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(1, self.ndim_problem))
            x[i] = np.clip(np.dot(x[i], self.sigma) + self.mean, self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def _update_parameters(self, mean, x, elite_x, y, elite_y):
        if self._n_generations == 0:
            order = np.argsort(y)
            for j in range(self.n_parents):
                elite_x[j] = x[order[j]]
                elite_y[j] = y[order[j]]
        else:
            length = int(self.fraction_reused_elites * self.n_parents)
            combine_x = np.empty((length, self.ndim_problem))
            combine_y = np.empty((length,))
            for k in range(length):
                combine_x[k] = elite_x[k]
                combine_y[k] = elite_y[k]
            combine_x = np.vstack((x, combine_x))
            combine_y = np.hstack((y, combine_y))
            order = np.argsort(combine_y)
            for j in range(self.n_parents):
                elite_x[j] = combine_x[order[j]]
                elite_y[j] = combine_y[order[j]]
        mean = self.alpha * mean + (1 - self.alpha) * np.mean(elite_x, axis=0)
        self.sigma = self.alpha * self.sigma + (1 - self.alpha) * np.std(elite_x, axis=0)
        return mean, elite_x, elite_y

    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, elite_x, y, elite_y = self.initialize()
        restart = True
        while True:
            self.n_individuals = int(max(self.n_individuals * np.power(self.decay, -1 * self._n_generations),
                                     2 * self.n_parents))
            if self.n_individuals != 2 * self.n_parents or restart is True:
                y = self.re_initialize()
                if self.n_individuals == 2 * self.n_parents:
                    restart = False
            x, y = self.iterate(mean, x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, elite_x, elite_y = self._update_parameters(mean, x, elite_x, y, elite_y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness, mean)
        return results
