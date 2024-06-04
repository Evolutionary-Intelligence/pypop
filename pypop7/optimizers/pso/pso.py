import numpy as np  # engine for numerical computing

from pypop7.optimizers.core.optimizer import Optimizer  # abstract class for all Black-Box Optimizers (BBO)


class PSO(Optimizer):
    """Particle Swarm Optimizer (PSO).

    This is the **abstract** class of all `PSO` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand. The unique goal of this abstract class is to unify the common
    interfaces of all its subclasses (different algorithm versions).

    .. note:: `PSO` is a very popular family of **swarm**-based search algorithms, originally proposed by an
       electrical engineer (Russell C. Eberhart) and a psychologist (James Kennedy), two recipients of `IEEE
       Evolutionary Computation Pioneer Award 2012 <https://tinyurl.com/456as566>`_. Its underlying motivation
       comes from interesting collective behaviors (e.g. `flocking <https://dl.acm.org/doi/10.1145/37402.37406>`_)
       observed in social animals (such as `birds <https://dl.acm.org/doi/10.1145/2629613>`_), which are often
       regarded as a particular form of *emergence* or *self-organization*. Recently, PSO-type swarm optimizers
       have been theoretically analyzed under the `Consensus-Based Optimization (CBO)
       <https://jmlr.csail.mit.edu/papers/v22/21-0259.html>`_ or `Swarm Gradient Dynamics
       <https://link.springer.com/article/10.1007/s10107-023-01988-8>`_ framework, with more or less
       modifications to the standard `PSO` implementation for mathematical tractability.

    For some interesting applications of `PSO`/`CBO` in diverse areas, please refer to `[Melis et al., 2024, Nature]
    <https://doi.org/10.1038/s41586-024-07293-4>`_, `[Wang et al., 2024, Nature Materials]
    <https://doi.org/10.1038/s41563-024-01830-2>`_, `[Elijošius et al., 2024]
    <https://doi.org/10.48550/arXiv.2402.08708>`_, `[Lugagne et al., 2024, Nature Communications]
    <https://doi.org/10.1038/s41467-024-46361-1>`_, `[Bottrell et al., MNRAS, 2024]
    <https://academic.oup.com/mnras/article/527/3/6506/7286659>`_, `[Chen et al., 2023, Nature Communications]
    <https://doi.org/10.1038/s41467-023-38185-2>`_, `[Guo et al., 2023, ISSTA]
    <https://dl.acm.org/doi/10.1145/3597926.3598058>`_, `[Yang et al., 2023, IEEE-TSP]
    <https://ieeexplore.ieee.org/abstract/document/10048578>`_, `[Liu et al., 2022, Nature Communications]
    <https://doi.org/10.1038/s41467-022-28083-4>`_, `[Benedetti et al., 2019]
    <https://www.nature.com/articles/s41534-019-0157-8>`_, `[Venter&Sobieszczanski-Sobieski, 2003, AIAAJ]
    <https://doi.org/10.2514/2.2111>`_, just to name a few.

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
    Bolte, J., Miclo, L. and Villeneuve, S., 2024.
    `Swarm gradient dynamics for global optimization: The mean-field limit case.
    <https://doi.org/10.1007/s10107-023-01988-8>`_
    Mathematical Programming, 205(1), pp.661-701.

    Cipriani, C., Huang, H. and Qiu, J., 2022.
    `Zero-inertia limit: From particle swarm optimization to consensus-based optimization.
    <https://doi.org/10.1137/21M1412323>`_
    SIAM Journal on Mathematical Analysis, 54(3), pp.3091-3121.

    Fornasier, M., Huang, H., Pareschi, L. and Sünnen, P., 2022.
    `Anisotropic diffusion in consensus-based optimization on the sphere.
    <https://doi.org/10.1137/21M140941X>`_
    SIAM Journal on Optimization, 32(3), pp.1984-2012.
 
    Fornasier, M., Huang, H., Pareschi, L. and Sünnen, P., 2021.
    `Consensus-based optimization on the sphere: Convergence to global minimizers and machine learning.
    <https://jmlr.csail.mit.edu/papers/v22/21-0259.html>`_
    Journal of Machine Learning Research, 22(1), pp.10722-10776.
 
    Blackwell, T. and Kennedy, J., 2018.
    `Impact of communication topology in particle swarm optimization.
    <https://doi.org/10.1109/TEVC.2018.2880894>`_
    IEEE Transactions on Evolutionary Computation, 23(4), pp.689-702.

    Bonyadi, M.R. and Michalewicz, Z., 2017.
    `Particle swarm optimization for single objective continuous space problems: A review.
    <https://doi.org/10.1162/EVCO_r_00180>`_
    Evolutionary Computation, 25(1), pp.1-54.

    https://www.cs.cmu.edu/~arielpro/15381f16/c_slides/781f16-26.pdf

    Floreano, D. and Mattiussi, C., 2008.
    `Bio-inspired artificial intelligence: Theories, methods, and technologies.
    <https://mitpress.mit.edu/9780262062718/bio-inspired-artificial-intelligence/>`_
    MIT Press.
    (See [Chapter 7.2 Particle Swarm Optimization] for details.)

    http://www.scholarpedia.org/article/Particle_swarm_optimization

    Poli, R., Kennedy, J. and Blackwell, T., 2007.
    `Particle swarm optimization.
    <https://doi.org/10.1007/s11721-007-0002-0>`_
    Swarm Intelligence, 1(1), pp.33-57.

    Clerc, M. and Kennedy, J., 2002.
    `The particle swarm-explosion, stability, and convergence in a multidimensional complex space.
    <https://doi.org/10.1109/4235.985692>`_
    IEEE Transactions on Evolutionary Computation, 6(1), pp.58-73.

    Eberhart, R.C., Shi, Y. and Kennedy, J., 2001.
    `Swarm intelligence.
    <https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4>`_
    Elsevier.

    Shi, Y. and Eberhart, R., 1998, May.
    `A modified particle swarm optimizer.
    <https://doi.org/10.1109/ICEC.1998.699146>`_
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.

    Kennedy, J. and Eberhart, R., 1995, November.
    `Particle swarm optimization.
    <https://doi.org/10.1109/ICNN.1995.488968>`_
    In Proceedings of International Conference on Neural Networks (pp. 1942-1948). IEEE.
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # swarm (population) size, aka number of particles
            self.n_individuals = 20
        self.cognition = options.get('cognition', 2.0)  # cognitive learning rate
        assert self.cognition >= 0.0
        self.society = options.get('society', 2.0)  # social learning rate
        assert self.society >= 0.0
        self.max_ratio_v = options.get('max_ratio_v', 0.2)  # maximal ratio of velocity
        assert 0.0 < self.max_ratio_v <= 1.0
        self.is_bound = options.get('is_bound', False)
        self._max_v = self.max_ratio_v*(self.upper_boundary - self.lower_boundary)  # maximal velocity
        self._min_v = -self._max_v  # minimal velocity
        self._topology = None  # neighbors topology of social learning
        self._n_generations = 0  # initial number of generations
        # set linearly decreasing inertia weights introduced in [Shi&Eberhart, 1998, IEEE-WCCI/CEC]
        self._max_generations = np.ceil(self.max_function_evaluations/self.n_individuals)
        if self._max_generations == np.Inf:
            self._max_generations = 1e2*self.ndim_problem
        self._w = 0.9 - 0.5*(np.arange(self._max_generations) + 1.0)/self._max_generations  # from 0.9 to 0.4
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
        return v, x, y, p_x, p_y, n_x

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y, n_x = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, y)
            v, x, y, p_x, p_y, n_x = self.iterate(v, x, y, p_x, p_y, n_x, args)
        return self._collect(fitness, y)
