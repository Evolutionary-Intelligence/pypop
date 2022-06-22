import numpy as np

from pypop7.optimizers.ga.ga import GA


def get_prob(prob, y1, y2):
    if y2 > y1:
        return_prob = prob + 0.1
        if return_prob > 0.95:
            return_prob = 0.95
    else:
        return_prob = prob - 0.1
        if return_prob < 0.05:
            return_prob = 0.05
    return return_prob


class GENITOR(GA):
    """GENITOR algorithm
    Reference
    --------------
    D. Whitley, S. Dominic, R. Das, C. W. Anderson
    Genetic Algorithm Learning for Neurocontrol Problems
    Machine Learning, 13, 259-284(1993)
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.options = options

    def initialize(self):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            x[i] = self._initialize_x()
            y[i] = self._evaluate_fitness(x[i])
        prob_cross = self.prob_cross * np.ones((self.n_individuals,))
        return x, y, prob_cross

    def iterate(self, x, y, prob_cross):
        order = np.argsort(y)
        rand = np.random.random()
        # do the crossover work
        if rand < prob_cross[order[0]]:
            x[order[-2]], x[order[-1]] = self.crossover(x[order[0]], x[order[1]], "one_point")
            y[order[-2]] = self._evaluate_fitness(x[order[-2]])
            y[order[-1]] = self._evaluate_fitness(x[order[-1]])
            prob_cross[order[-2]] = get_prob(prob_cross[order[0]], y[order[-2]], y[order[0]])
            prob_cross[order[-1]] = get_prob(prob_cross[order[0]], y[order[-1]], y[order[0]])
        # do the mutate work
        else:
            x[order[0]] = self.mutate(x[order[0]])
            y[order[0]] = self._evaluate_fitness(x[order[0]])
        return x, y, prob_cross

    def mutate(self, x):
        for i in range(self.ndim_problem):
            rand = np.random.random()
            if rand < self.prob_mutate:
                x[i] = x[i] + np.random.random() * 20 - 10
        return x

    def optimize(self, fitness_function=None):
        fitness = GA.optimize(self, fitness_function)
        x, y, prob_cross = self.initialize()
        while True:
            x, y, prob_cross = self.iterate(x, y, prob_cross)
            if self.record_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self.__collect_results(fitness)
        return results
