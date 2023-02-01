import numpy as np

from pypop7.optimizers.es.es import ES


class FCMAES(ES):
    """Fast Covariance Matrix Adaptation Evolution Strategy (FCMAES).

    References
    ----------
    Li, Z., Zhang, Q., Lin, X. and Zhen, H.L., 2020.
    Fast covariance matrix adaptation for large-scale black-box optimization.
    IEEE Transactions on Cybernetics, 50(5), pp.2073-2083.
    https://ieeexplore.ieee.org/abstract/document/8533604

    Li, Z. and Zhang, Q., 2016.
    What does the evolution path learn in CMA-ES?.
    In Parallel Problem Solving from Nature (pp. 751-760).
    Springer International Publishing.
    https://link.springer.com/chapter/10.1007/978-3-319-45823-6_70
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.m = self.n_individuals  # number of evolution paths
        self.c = 2.0/(self.ndim_problem + 5.0)  # learning rate of evolution path update
        self.c_1 = 1.0/(3.0*np.sqrt(self.ndim_problem) + 5.0)  # sampling factor
        self.c_s = 0.3  # learning rate of rank-based success rule for global step-size adaptation
        self.q_star = 0.27  # target of rank-based success rule for global step-size adaptation
        self.d_s = 1.0  # damping factor of rank-based success rule for global step-size adaptation
        self.n_steps = self.ndim_problem  # updating frequency of direction vector set
        self._x_1 = 1.0 - self.c_1
        self._x_2 = np.sqrt((1.0 - self.c_1)*self.c_1)
        self._x_3 = np.sqrt(self.c_1)
        self._p_1 = 1.0 - self.c
        self._p_2 = None
        self._rr = None

    def initialize(self, is_restart=False):
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        p = np.zeros((self.ndim_problem,))  # evolution path
        p_hat = np.zeros((self.m, self.ndim_problem))  # direction vector set
        s = 0
        self._p_2 = np.sqrt(self.c*(2.0 - self.c)*self._mu_eff)
        self._rr = np.arange(self.n_parents*2) + 1
        return mean, x, y, p, p_hat, s

    def iterate(self, mean=None, x=None, y=None, p=None, p_hat=None, args=None):
        for i in range(self.n_individuals):
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            if self._n_generations < self.m:  # unbiased sampling when starting
                x[i] = mean + self.sigma*z
            else:
                x[i] = mean + self.sigma*(self._x_1*z +
                                          self._x_2*self.rng_optimization.standard_normal()*p_hat[i] +
                                          self._x_3*self.rng_optimization.standard_normal()*p)
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def _update_distribution(self, mean=None, x=None, y=None, p=None, p_hat=None, s=None, y_bak=None):
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_bak = np.dot(self._w[:self.n_parents], x[order])
        p = self._p_1*p + self._p_2*(mean_bak - mean)/self.sigma
        if self._n_generations % self.n_steps == 0:
            p_hat[:-1] = p_hat[1:]
            p_hat[-1] = p
        if self._n_generations > 0:
            r = np.argsort(np.hstack((y_bak[:self.n_parents], y[:self.n_parents])))
            rr = self._rr[r < self.n_parents] - self._rr[r >= self.n_parents]
            q = np.dot(self._w, rr)/self.n_parents
            s = (1.0 - self.c_s)*s + self.c_s*(q - self.q_star)
            self.sigma *= np.exp(s/self.d_s)
        self._n_generations += 1
        return mean_bak, p, p_hat, s

    def restart_reinitialize(self, mean=None, x=None, y=None, p=None, p_hat=None, s=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            self.d_s *= 2.0
            self.m = self.n_individuals
            mean, x, y, p, p_hat, s = self.initialize(True)
        return mean, x, y, p, p_hat, s

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        mean, x, y, p, p_hat, s = self.initialize()
        while not self._check_terminations():
            y_bak = np.copy(y)
            x, y = self.iterate(mean, x, y, p, p_hat, args)
            self._print_verbose_info(fitness, y)
            mean, p, p_hat, s = self._update_distribution(mean, x, y, p, p_hat, s, y_bak)
            mean, x, y, p, p_hat, s = self.restart_reinitialize(mean, x, y, p, p_hat, s)
        results = self._collect(fitness, y, mean)
        results['p'] = p
        results['s'] = s
        return results
