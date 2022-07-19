import numpy as np

from pypop7.optimizers.cc.cc import CC


class COSYNE(CC):
    """Cooperative Synapse Neuroevolution(CoSyNE)
    Reference
    -----------
    F. Gomez, J. Schmidhuber, R. Miikkulainen
    Accelerated Neural Evolution through Cooperatively Coevolved Synapses
    https://jmlr.org/papers/v9/gomez08a.html
    """
    def __init__(self, problem, options):
        CC.__init__(self, problem, options)
        self.n_combine = options.get('n_combine')
        self.crossover_type = options.get('crossover_type')
        self.prob_mutate = options.get('prob_mutate')

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x()
        return x, y

    def mutate(self, x):
        for i in range(2):
            for j in range(self.ndim_problem):
                rand = np.random.random()
                if rand < self.prob_mutate:
                    x[i][j] = x[i][j] + 0.3 * self.rng_optimization.standard_cauchy(1)
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
        return x

    def permute(self, log):
        temp = []
        length = len(log)
        for k in range(length):
            index = np.random.randint(0, len(log)) % len(log)
            temp.append(log[index])
            log.pop(index)
        return temp

    def iterate(self, x, y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[i] = self._evaluate_fitness(x[i])
        order = np.argsort(y)
        o = self.crossover(x[order[0]], x[order[1]], self.crossover_type)
        o = self.mutate(o)
        for i in range(self.ndim_problem):
            p = x[:, i]
            order_1 = np.argsort(p)
            for k in range(2):
                x[order_1[self.n_individuals - k - 1]][i] = o[k][i]
            mark_weight, log = [], []
            for j in range(self.n_individuals):
                temp = (y[j] - y[order[0]])/(y[order[-1]] - y[order[0]])
                prob = np.power(temp, 1.0 / self.ndim_problem)
                rand = np.random.random()
                if rand < prob:
                    mark_weight.append(x[j][i])
                    log.append(j)
            log = self.permute(log)
            for j in range(len(log)):
                x[log[j]][i] = mark_weight[j]
        return x, y

    def optimize(self, fitness_function=None):
        fitness = CC.optimize(self, fitness_function)
        x, y = self.initialize()
        while True:
            x, y = self.iterate(x, y)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self.__collect_results(fitness)
        return results
