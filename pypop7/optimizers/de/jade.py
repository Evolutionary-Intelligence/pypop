import numpy as np

from pypop7.optimizers.de.de import DE


class JADE(DE):
    """Adaptive Differential Evolution (JADE).

    Reference
    ---------
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), pp.945–958.
    https://doi.org/10.1109/TEVC.2009.2014613
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.mu = options.get('mu', 0.5)  # The initial mean of normal distribution
        self.median = options.get('median', 0.5)  # The initial median of the Cauchy distribution
        self.s = options.get('s', 0.1)  # The scale parameter of the Cauchy distribution
        self.p = options.get('p', 0.05)  # p determines the greediness of the mutation strategy
        self.c = options.get('c', 0.1)  # 1/c is the life span of a successful crossover probability or mutation factor
        self.archive = options.get('archive', True)  # Whether archive the inferior solution
        self.boundary = options.get('boundary', False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            (self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))  # the set of archived inferior solutions
        self._n_generations += 1
        return x, y, a

    def bound(self, x=None, xx=None):
        if not self.boundary:
            return x
        for k in range(self.n_individuals):
            idx = np.array(x[k] < self.lower_boundary)
            if idx.any():
                x[k][idx] = (self.lower_boundary + xx[k])[idx] / 2
            idx = np.array(x[k] > self.upper_boundary)
            if idx.any():
                x[k][idx] = (self.upper_boundary + xx[k])[idx] / 2
        return x

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals,  self.ndim_problem))
        f_mu = np.empty((self.n_individuals,))  # mutation factors
        order = np.argsort(y)[:int(np.ceil(self.p * self.n_individuals))]  # Choose the 100p% best individuals
        x_p = x[self.rng_optimization.choice(order, (self.n_individuals,))]
        x_un = np.copy(x)
        if self.archive:
            x_un = np.vstack((x_un, a))  # The union of the current population and the archive
        for k in range(self.n_individuals):
            f_mu[k] = self.s * self.rng_optimization.standard_cauchy() + self.median
            while f_mu[k] <= 0:
                f_mu[k] = self.s * self.rng_optimization.standard_cauchy() + self.median
            if f_mu[k] > 1:
                f_mu[k] = 1
            r1 = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), k))
            r2 = self.rng_optimization.choice(np.setdiff1d(np.arange(len(x_un)), np.union1d(k, r1)))
            x_mu[k] = x[k] + f_mu[k] * (x_p[k] - x[k]) + f_mu[k] * (x[r1] - x_un[r2])
        return x_mu, f_mu

    def crossover(self, x_mu=None, x=None):
        x_cr = np.copy(x)
        p_cr = self.rng_optimization.normal(self.mu, 0.1, (self.n_individuals,))  # crossover probabilities
        for k in range(self.n_individuals):
            for i in range(self.ndim_problem):
                if (i == self.rng_optimization.integers(self.ndim_problem)) or\
                        (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f = np.empty((0,))  # the set of all successful mutation factors
        p = np.empty((0,))  # the set of all successful crossover probabilities
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive the inferior solution
                f = np.hstack((f, f_mu[k]))  # archive the successful mutation factor
                p = np.hstack((p, p_cr[k]))  # archive the successful crossover probability
                x[k] = x_cr[k]
                y[k] = yy
        if len(p) != 0:
            self.mu = (1 - self.c) * self.mu + self.c * np.mean(p)  # Update the mean of Normal distribution
        if len(f) != 0:  # Update the location of Cauchy distribution
            self.median = (1 - self.c) * self.median + self.c * np.sum(np.power(f, 2)) / np.sum(f)
        return x, y, a

    def iterate(self, args=None, x=None, y=None, a=None):
        x_mu, f_mu = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        if len(a) > self.n_individuals:  # randomly remove solutions from a so that |a| <= self.n_individuals
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, a = self.initialize(args)
        fitness.extend(y)
        while True:
            x, y, a = self.iterate(args, x, y, a)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
