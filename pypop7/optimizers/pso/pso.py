import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class PSO(Optimizer):
    """Particle Swarm Optimizer (PSO).

    This is the **base** (abstract) class for all `PSO` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

    .. note:: `PSO` is a very popular family of *swarm*-based search algorithms, proposed together by an electrical
       engineer (Russell C. Eberhart) and a psychologist (James Kennedy), recipients of `Evolutionary Computation
       Pioneer Award 2012 <https://tinyurl.com/456as566>`_. Its underlying motivation comes from very interesting
       collective behaviors (e.g. `flocking <https://dl.acm.org/doi/10.1145/37402.37406>`_) observed from social
       animals (such as `birds <https://dl.acm.org/doi/10.1145/2629613>`_), which are regarded as one particular form
       of *intelligence* or *emergence* or *self-organization* by many scientists and engineers.

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
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, aka number of particles (`int`, default: `20`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.0`),
                * 'society'       - social learning rate (`float`, default: `2.0`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Attributes
    ----------
    cognition     : `float`
                    cognitive learning rate, aka acceleration coefficient.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.
    society       : `float`
                    social learning rate, aka acceleration coefficient.

    Methods
    -------

    References
    ----------
    Blackwell, T. and Kennedy, J., 2018.
    Impact of communication topology in particle swarm optimization.
    IEEE Transactions on Evolutionary Computation, 23(4), pp.689-702.
    https://ieeexplore.ieee.org/abstract/document/8531770

    Bonyadi, M.R. and Michalewicz, Z., 2017.
    Particle swarm optimization for single objective continuous space problems: A review.
    Evolutionary Computation, 25(1), pp.1-54.
    https://direct.mit.edu/evco/article-abstract/25/1/1/1040/Particle-Swarm-Optimization-for-Single-Objective

    https://www.cs.cmu.edu/~arielpro/15381f16/c_slides/781f16-26.pdf

    Floreano, D. and Mattiussi, C., 2008.
    Bio-inspired artificial intelligence: Theories, methods, and technologies.
    MIT Press.
    https://mitpress.mit.edu/9780262062718/bio-inspired-artificial-intelligence/
    (See [Chapter 7.2 Particle Swarm Optimization] for details.)

    http://www.scholarpedia.org/article/Particle_swarm_optimization

    Poli, R., Kennedy, J. and Blackwell, T., 2007.
    Particle swarm optimization.
    Swarm Intelligence, 1(1), pp.33-57.
    https://link.springer.com/article/10.1007/s11721-007-0002-0

    Venter, G. and Sobieszczanski-Sobieski, J., 2003.
    Particle swarm optimization.
    AIAA Journal, 41(8), pp.1583-1589.
    https://arc.aiaa.org/doi/abs/10.2514/2.2111

    Clerc, M. and Kennedy, J., 2002.
    The particle swarm-explosion, stability, and convergence in a multidimensional complex space.
    IEEE Transactions on Evolutionary Computation, 6(1), pp.58-73.
    https://ieeexplore.ieee.org/abstract/document/985692

    Eberhart, R.C., Shi, Y. and Kennedy, J., 2001.
    Swarm intelligence.
    Elsevier.
    https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4

    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146

    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # swarm (population) size, aka number of particles
            self.n_individuals = 20
        self.cognition = options.get('cognition', 2.0)  # cognitive learning rate
        self.society = options.get('society', 2.0)  # social learning rate
        self.max_ratio_v = options.get('max_ratio_v', 0.2)  # maximal ratio of velocity
        self._max_v = self.max_ratio_v*(self.upper_boundary - self.lower_boundary)  # maximal velocity
        self._min_v = -self._max_v  # minimal velocity
        self._topology = None  # neighbors topology of social learning
        self._n_generations = 0  # number of generations
        # set linearly decreasing inertia weights introduced in [Shi&Eberhart, 1998, WCCI]
        self._max_generations = np.ceil(self.max_function_evaluations / self.n_individuals)
        if self._max_generations == np.Inf:
            self._max_generations = 1e2 * self.ndim_problem
        self._w = 0.9 - 0.5 * (np.arange(self._max_generations) + 1.0) / self._max_generations  # from 0.9 to 0.4
        self._swarm_shape = (self.n_individuals, self.ndim_problem)

    def initialize(self, args=None):
        v = self.rng_initialization.uniform(self._min_v, self._max_v, size=self._swarm_shape)  # velocities
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=self._swarm_shape)  # positions
        y = np.empty((self.n_individuals,))  # fitness
        p_x, p_y = np.copy(x), np.copy(y)  # personally previous-best positions and fitness
        n_x = np.copy(x)  # neighborly previous-best positions
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            y[i] = self._evaluate_fitness(x[i], args)
        p_y = np.copy(y)
        return v, x, y, p_x, p_y, n_x

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        self._n_generations += 1
        raise NotImplementedError

    def _print_verbose_info(self, y=None):
        if self.verbose and (not self._n_generations % self.verbose):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y, n_x = self.initialize(args)
        if self.saving_fitness:
            fitness.extend(y)
        self._print_verbose_info(y)
        while True:
            v, x, y, p_x, p_y, n_x = self.iterate(v, x, y, p_x, p_y, n_x, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
        return self._collect_results(fitness)
