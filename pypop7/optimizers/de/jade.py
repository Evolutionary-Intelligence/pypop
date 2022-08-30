import numpy as np

from pypop7.optimizers.de.de import DE


class JADE(DE):
    """Adaptive Differential Evolution (JADE).

    Reference
    ---------
    Zhang, J., and Sanderson, A. C. 2009.
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945–958.
    https://doi.org/10.1109/TEVC.2009.2014613
    """
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.mu = options.get('mu', 0.5)  # The initial mean of normal distribution
        self.median = options.get('median', 0.5)  # The initial location parameter of the Cauchy distribution
        self.s = options.get('s', 0.1)  # The scale parameter of the Cauchy distribution
        self.p = options.get('p', 0.05)  # p determines the greediness of the mutation strategy
        self.c = options.get('c', 0.1)  # 1/c is the life span of a successful crossover probability or mutation factor
        self.archive = options.get('archive', True)
        self.boundary_constraint = options.get('boundary_constraint', False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))  # the set of archived inferior solutions
        self._n_generations += 1
        return x, y, a

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals,  self.ndim_problem))
        mu_f = np.empty((self.n_individuals,))  # mutation factors
        order = np.argsort(y)[:int(np.ceil(self.p * self.n_individuals))]  # Choose the 100p% best individuals
        x_p = x[self.rng_optimization.choice(order, (self.n_individuals,))]
        x_un = np.copy(x)
        if self.archive:
            x_un = np.vstack((x_un, a))  # The union of the current population and the archive
        for i in range(self.n_individuals):
            mu_f[i] = self.s * self.rng_optimization.standard_cauchy() + self.median
            while mu_f[i] <= 0:
                mu_f[i] = self.s * self.rng_optimization.standard_cauchy() + self.median
            if mu_f[i] > 1:
                mu_f[i] = 1
            r1 = self.rng_optimization.choice(np.setdiff1d(np.arange(self.n_individuals), i))
            r2 = self.rng_optimization.choice(np.setdiff1d(np.arange(len(x_un)), np.union1d(i, r1)))
            x_mu[i] = x[i] + mu_f[i] * (x_p[i] - x[i]) + mu_f[i] * (x[r1] - x_un[r2])

            if self.boundary_constraint:
                idx = np.array(x_mu[i] < self.lower_boundary)
                if idx.any():
                    x_mu[i][idx] = (self.lower_boundary[idx] + x[i][idx]) / 2
                idx = np.array(x_mu[i] > self.upper_boundary)
                if idx.any():
                    x_mu[i][idx] = (self.upper_boundary[idx] + x[i][idx]) / 2
        return x_mu, mu_f

    def crossover(self, x=None, x_mu=None):
        cr_p = self.rng_optimization.normal(self.mu, 0.1, (self.n_individuals,))  # crossover probabilities
        x_cr = np.copy(x)
        for i in range(self.n_individuals):
            for j in range(self.ndim_problem):
                if (j == self.rng_optimization.integers(self.ndim_problem)) or \
                        (self.rng_optimization.random() < cr_p[i]):
                    x_cr[i, j] = x_mu[i, j]
        return x_cr, cr_p

    def select(self, args=None, x=None, y=None, a=None, x_cr=None, mu_f=None, cr_p=None):
        s_f = []  # the set of all successful mutation factors
        s_p = []  # the set of all successful crossover probabilities
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y_temp = self._evaluate_fitness(x_cr[i], args)
            if y_temp < y[i]:
                x[i] = x_cr[i]
                y[i] = y_temp
                if self.archive:
                    a = np.vstack((a, x_cr[i]))
                s_f.append(mu_f[i])
                s_p.append(cr_p[i])
        return x, y, a, s_f, s_p

    def iterate(self, args=None, x=None, y=None, a=None):
        x_mu, mu_f = self.mutate(x, y, a)
        x_cr, cr_p = self.crossover(x, x_mu)
        x, y, a, s_f, s_p = self.select(args, x, y, a, x_cr, mu_f, cr_p)

        if self.archive:
            if len(a) > self.n_individuals:  # randomly remove solutions from a so that |a| <= self.n_individuals
                a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        self.mu = (1 - self.c) * self.mu + self.c * np.mean(s_p)  # Update the mean of Normal distribution
        if np.sum(s_f) != 0:  # Update the location of Cauchy distribution
            self.median = (1 - self.c) * self.median + self.c * np.sum(np.power(s_f, 2)) / np.sum(s_f)
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
        return self._collect_results(fitness)
