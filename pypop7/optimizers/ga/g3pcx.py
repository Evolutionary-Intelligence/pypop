import numpy as np

from pypop7.optimizers.ga.ga import GA


class G3PCX(GA):
    """G3PCX

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).

    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`),
              and with the following particular settings (`keys`):
                * 'n_individuals' - population size (`int`, default: `100`),
                * 'n_parents' - parent size (`int`, default: `3`),
                * 'n_offsprings' - offspring size (`int`, default: `2`),
                * 'n_family' - family size (`int`, default: `2`),
                * 'sigma_eta' - distributed variable of PCA (`float`, default: `0.1`),
                * 'sigma_zeta' - distributed variable of PCA (`float`, default: `0.1`),
                * 'mut_eta' - distributed variable of PM (`float`, default: `0.1`),

    Examples
    --------
    Use the GA optimizer `G3PCX` to minimize the well-known test function
    `Rastrigin <http://en.wikipedia.org/wiki/Rastrigin_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rastrigin  # function to be minimized
       >>> from pypop7.optimizers.ga.g3pcx import G3PCX
       >>> problem = {'fitness_function': rastrigin,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5 * numpy.ones((100,)),
       ...            'upper_boundary': 5 * numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'n_individuals': 100,
       ...            'n_parents': 3,
       ...            'seed_rng': 2022}
       >>> g3pcx = G3PCX(problem, options)  # initialize the optimizer class
       >>> results = g3pcx.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"G3PCX: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       G3PCX: 185180, 9.208633855450898e-11

    Attributes
    ----------
    n_individuals : `int`
                    population size.
    n_parents : `int`
                    parent size.
    n_offsprings : `int`
                    offspring size.
    n_family : `int`
                    family size.
    sigma_eta : `float`
                    distributed variable of PCA.
    sigma_zeta : `float`
                    distributed variable of PCA.
    mut_eta : `float`
                    distributed variable of PM.

    References
    ----------
    Kalyanmoy Deb, Ashish Anand, and Dhiraj Joshi.
    A computationally efficient evolutionary algorithm for real-parameter optimization.
    Evolutionary Computation, 10(4):371–395, 2002.
    Part of the code is referenced from:
    https://pymoo.org/algorithms/soo/g3pcx.html
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.n_offsprings = options.get('n_offsprings', 2)
        self.n_family = options.get('n_family', 2)
        self.sigma_eta = options.get('sigma_eta', 0.1)
        self.sigma_zeta = options.get('sigma_zeta', 0.1)
        self.mut_eta = options.get('mut_eta', 20)
        self.EPSILON = 1e-32
        self.sp_size = self.n_offsprings + self.n_family
        self.indices = np.arange(0, self.n_individuals)
        self.prob_var = min(0.5, 1 / self.ndim_problem)
        self.best_index = None

    def initialize(self, args):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            y[i] = self._evaluate_fitness(x[i], args)
        self.best_index = np.argmin(y)
        return x, y

    def parent_centric_xover(self, x=None, index=None):
        # calculate the differences from all parents to index parent
        diff_to_index = x - x[index]
        dist_to_index = np.linalg.norm(diff_to_index, axis=-1)
        dist_to_index = np.maximum(self.EPSILON, dist_to_index)

        # find the centroid of the parents
        centroid = np.mean(x, axis=0)

        # calculate the difference between the centroid and the k-th parent
        diff_to_centroid = centroid - x[index]

        dist_to_centroid = np.linalg.norm(diff_to_centroid, axis=-1)
        dist_to_centroid = np.maximum(self.EPSILON, dist_to_centroid)

        # orthogonal directions are computed
        orth_dir = np.zeros_like(dist_to_index)

        for i in range(self.n_parents):
            if i != index:
                temp1 = (diff_to_index[i] * diff_to_centroid).sum(axis=-1)
                temp2 = temp1 / (dist_to_index[i] * dist_to_centroid)
                temp3 = np.maximum(0.0, 1.0 - temp2 ** 2)
                orth_dir[i] = dist_to_index[i] * (temp3 ** 0.5)

        # this is the avg of the perpendicular distances from other parents to the parent k
        D_not = orth_dir.sum(axis=0) / (self.n_parents - 1)
        # generating zero-mean normally distributed variables
        sigma = D_not * np.repeat(self.sigma_eta, self.ndim_problem)
        rnd = np.random.normal(loc=0.0, scale=sigma)

        # implemented just like the c code - generate_new.h file
        inner_prod = np.sum(rnd * diff_to_centroid, axis=-1, keepdims=True)
        noise = rnd - (inner_prod * diff_to_centroid) / dist_to_centroid ** 2

        bias_to_centroid = np.random.normal(0.0, self.sigma_zeta) * diff_to_centroid

        # the array which is finally returned
        return x[index] + noise + bias_to_centroid

    def polynomial_mutation(self, x):
        Xp = x
        mut = np.random.random(self.ndim_problem) < self.prob_var
        mut[self.lower_boundary == self.upper_boundary] = False

        _xl = self.lower_boundary[mut]
        _xu = self.upper_boundary[mut]

        x1 = x[mut]
        eta = np.tile(self.mut_eta, self.ndim_problem)[mut]

        delta1 = (x1 - _xl) / (_xu - _xl)
        delta2 = (_xu - x1) / (_xu - _xl)

        mut_pow = 1.0 / (eta + 1.0)
        rand = np.random.random(x1.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(x1.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = x1 + deltaq * (_xu - _xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < _xl] = _xl[_Y < _xl]
        _Y[_Y > _xu] = _xu[_Y > _xu]

        # set the values for output
        Xp[mut] = _Y
        return Xp

    def iterate(self, x=None, y=None, fitness=None):
        indexs = np.random.choice(self.indices, self.n_parents)
        if self.best_index not in indexs:
            indexs[0] = self.best_index

        offsprings = np.empty((self.sp_size, self.ndim_problem))
        off_y = np.zeros((self.sp_size,))
        parent_x = np.empty((self.n_parents, self.ndim_problem))
        for i in range(self.n_parents):
            parent_x[i] = x[indexs[i]]
        parents = np.arange(0, self.n_individuals)

        for i in range(self.n_offsprings):
            offsprings[i] = self.parent_centric_xover(parent_x, i)
            offsprings[i] = np.clip(offsprings[i], self.lower_boundary, self.upper_boundary)
            offsprings[i] = self.polynomial_mutation(offsprings[i])
            offsprings[i] = np.clip(offsprings[i], self.lower_boundary, self.upper_boundary)
            off_y[i] = self._evaluate_fitness(offsprings[i])

        for i in range(self.n_family):
            rindex = np.random.randint(0, self.n_individuals - 1) + i
            if rindex > self.n_individuals - 1:
                rindex = self.n_individuals - 1
            temp = parents[rindex]
            parents[rindex] = parents[i]
            parents[i] = temp
        for i in range(self.n_family):
            offsprings[self.n_offsprings + i] = x[parents[i]]
            off_y[self.n_offsprings + i] = y[parents[i]]
        if self.saving_fitness:
            fitness.extend(off_y)

        order = np.argsort(off_y)
        for i in range(self.n_family):
            x[parents[i]] = offsprings[order[i]]
            y[parents[i]] = off_y[order[i]]
        self.best_index = np.argmin(y)
        return x, y, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y = self.initialize(args=None)
        fitness.extend(y)
        while True:
            x, y, fitness = self.iterate(x, y, fitness)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
            self._n_generations += 1
        return self._collect_results(fitness)
