import numpy as np
import numba as nb

from pypop7.optimizers.es.es import ES


@nb.jit(nopython=True)
def cholesky_update(rm, z, downdate):
    # https://github.com/scipy/scipy/blob/d20f92fce9f1956bfa65365feeec39621a071932/
    #     scipy/linalg/_decomp_cholesky_update.py
    rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1., 1.
    sign = -1 if downdate else 1
    for r in range(len(z)):
        a = z[r]/rm[r, r]
        alpha[r] = alpha[r - 1] + sign*np.power(a, 2)
        beta[r] = np.sqrt(alpha[r])
        z[r + 1:] -= a*rm[r, r + 1:]
        rm[r, r:] *= beta[r]/beta[r - 1]
        rm[r, r + 1:] += sign*a/(beta[r]*beta[r - 1])*z[r + 1:]
    return rm.T


class OPOA2015(ES):
    """(1+1)-Active-CMA-ES 2015 (OPOA2015).

    Krause, O. and Igel, C., 2015, January.
    A more efficient rank-one covariance matrix update for evolution strategies.
    In Proceedings of ACM Conference on Foundations of Genetic Algorithms (pp. 129-136).
    https://dl.acm.org/doi/abs/10.1145/2725494.2725496
    """
    def __init__(self, problem, options):
        options['n_individuals'] = 1  # mandatory setting for OPOA2015
        options['n_parents'] = 1  # mandatory setting for OPOA2015
        ES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/(1.0 + self.ndim_problem/2.0)
        self.p_ts = options.get('p_ts', 2.0/11.0)
        self.c_p = options.get('c_p', 1.0/12.0)
        self.c_c = options.get('c_c', 2.0/(self.ndim_problem + 2.0))
        self.c_cov = options.get('c_cov', 2.0/(np.power(self.ndim_problem, 2) + 6.0))
        self.p_t = options.get('p_t', 0.44)
        self.c_m = options.get('c_m', 0.4/(np.power(self.ndim_problem, 1.6) + 1.0))
        self.k = options.get('k', 5)
        self._ancestors = []
        self._c_cf = 1.0 - self.c_cov + self.c_cov*self.c_c*(2.0 - self.c_c)

    def initialize(self, args=None, is_restart=False):
        mean = self._initialize_mean(is_restart)
        y = self._evaluate_fitness(mean, args)  # fitness
        cf = np.diag(np.ones(self.ndim_problem,))  # Cholesky factorization
        best_so_far_y, p_s = np.copy(y), self.p_ts
        p_c = np.zeros((self.ndim_problem,))  # evolution path
        return mean, y, cf, best_so_far_y, p_s, p_c

    def _cholesky_update(self, cf=None, alpha=None, beta=None, v=None):  # triangular rank-one update
        assert self.ndim_problem == v.size
        if beta < 0:
            downdate, beta = True, -beta
        else:
            downdate = False
        return cholesky_update(np.sqrt(alpha)*cf, np.sqrt(beta)*v, downdate)

    def iterate(self, args=None, mean=None, cf=None, best_so_far_y=None, p_s=None, p_c=None):
        # sample and evaluate only one offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        cf_z = np.dot(cf, z)
        x = mean + self.sigma*cf_z
        y = self._evaluate_fitness(x, args)
        if y <= best_so_far_y:
            self._ancestors.append(y)
            mean, best_so_far_y = x, y
            p_s = (1.0 - self.c_p)*p_s + self.c_p
            is_better = True
        else:
            p_s *= 1.0 - self.c_p
            is_better = False
        self.sigma *= np.exp(self.lr_sigma*(p_s - self.p_ts)/(1.0 - self.p_ts))
        if p_s >= self.p_t:
            p_c *= 1.0 - self.c_c
            cf = self._cholesky_update(cf, self._c_cf, self.c_cov, p_c)
        elif is_better:
            p_c = (1.0 - self.c_c)*p_c + np.sqrt(self.c_c*(2.0 - self.c_c))*cf_z
            cf = self._cholesky_update(cf, 1.0 - self.c_cov, self.c_cov, p_c)
        elif len(self._ancestors) >= self.k and y > self._ancestors[-self.k]:
            del self._ancestors[0]
            c_m = np.minimum(self.c_m, 1.0/(2.0*np.dot(z, z) - 1.0))
            cf = self._cholesky_update(cf, 1.0 + c_m, -c_m, cf_z)
        return mean, cf, best_so_far_y, p_s, p_c, y

    def restart_reinitialize(self, args=None, mean=None, y=None, cf=None,
                             best_so_far_y=None, p_s=None, p_c=None, fitness=None):
        self._fitness_list.append(self.best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._fitness_list) >= self.stagnation:
            is_restart_2 = (self._fitness_list[-self.stagnation] - self._fitness_list[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._n_restart += 1
            self.sigma = np.copy(self._sigma_bak)
            mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(args, True)
            if self.saving_fitness:
                fitness.append(y)
            self._fitness_list = [best_so_far_y]
            self._ancestors = []
            self._n_generations = 0
            if self.verbose:
                print(' ....... restart .......')
            self._print_verbose_info(y)
        return mean, y, cf, best_so_far_y, p_s, p_c

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(args)
        if self.saving_fitness:
            fitness.append(y)
        self._print_verbose_info(y)
        while True:
            mean, cf, best_so_far_y, p_s, p_c, y = self.iterate(
                args, mean, cf, best_so_far_y, p_s, p_c)
            if self.saving_fitness:
                fitness.append(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                mean, y, cf, best_so_far_y, p_s, p_c = self.restart_reinitialize(
                    args, mean, y, cf, best_so_far_y, p_s, p_c, fitness)
        return self._collect_results(fitness, mean)
