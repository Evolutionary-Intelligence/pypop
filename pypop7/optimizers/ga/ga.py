import numpy as np  # engine for numerical computing

from pypop7.optimizers.core.optimizer import Optimizer  # abstract class of all Black-Box Optimizers (BBO) classes


class GA(Optimizer):
    """Genetic Algorithms (GA).

    This is the **abstract** class for all `GA` classes. Please use any of its instantiated subclasses to
    optimize the **black-box** problem at hand.

    .. note:: `GA` are one of three *earliest* versions of evolutionary algorithms along with *evolutionary programming*
       (`EP`) and *evolution strategies* (`ES`). GA' original history dated back to Holland's **JACM** paper in 1962
       called *outline for a logical theory of adaptive systems*. **John H. Holland**, "GA's father", was the 2003
       recipient of `IEEE Evolutionary Computation Pioneer Award <https://tinyurl.com/456as566>`_. Note that both Hans
       Bremermann (professor emeritus at University of California at Berkeley) and Woody Bledsoe (chairman in IJCAI-1977
       / president-elect in AAAI-1983) did independent works closest to the modern notion of `GA`, as pointed out by
       `[Goldberg, 1989] <https://www.goodreads.com/en/book/show/142613>`_.

       *"Just to give you a flavor of these problems: GA have been used at the General Electric Company for automating
       parts of aircraft design, Los Alamos National Lab for analyzing satellite images, the John Deere company for
       automating assembly line scheduling, and Texas Instruments for computer chip design. GA were used for generating
       realistic computer-animated horses in the 2003 movie The Lord of the Rings: The Return of the King, and realistic
       computer-animated stunt doubles for actors in the movie Troy. A number of pharmaceutical companies are using GA
       to aid in the discovery of new drugs. GA have been used by several financial organizations for various tasks:
       detecting fraudulent trades (London Stock Exchange), analysis of credit card data (Capital One), and forecasting
       financial markets and portfolio optimization (First Quadrant). In the 1990s, collections of artwork created by an
       interactive GA were exhibited at several museums, including the Georges Pompidou Center in Paris. These examples
       are just a small sampling of ways in which GA are being used."*---`[Mitchell, 2009, 《Complexity: A Guided Tour》
       --winner of the 2010 Phi Beta Kappa Book Award in Science]
       <https://www.amazon.com/Complexity-Guided-Tour-Melanie-Mitchell/dp/0199798109>`_

    For some interesting applications of `GA` on diverse areas, please refer to `[Lyu et al., 2024, Science]
    <https://www.science.org/doi/10.1126/science.adn6354>`_, `[Truong-Quoc et al., 2024, Nature Materials]
    <https://www.nature.com/articles/s41563-024-01846-8>`_, `[Castanha et al., 2024, PNAS]
    <https://www.pnas.org/doi/abs/10.1073/pnas.2312755121>`_, `[Lucas et al., 2023, Nature Photonics]
    <https://www.nature.com/articles/s41566-023-01252-7>`_, `[Villard et al., 2023, JCTC]
    <https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c01078>`_, `[Kanal&Hutchison, 2017]
    <https://arxiv.org/abs/1707.02949>`_, `[Groenendaal et al., 2015, PLoS Computational Biology]
    <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004242>`_, to name a few.

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular setting (`key`):
                * 'n_individuals' - population size (`int`, default: `100`).

    Attributes
    ----------
    n_individuals : `int`
                    population size.

    Methods
    -------

    References
    ----------
    Whitley, D., 2019.
    `Next generation genetic algorithms: A user’s guide and tutorial.
    <https://link.springer.com/chapter/10.1007/978-3-319-91086-4_8>`_
    In Handbook of Metaheuristics (pp. 245-274). Springer, Cham.

    De Jong, K.A., 2006.
    `Evolutionary computation: A unified approach.
    <https://mitpress.mit.edu/9780262041942/evolutionary-computation/>`_
    MIT Press.

    Mitchell, M., 1998.
    `An introduction to genetic algorithms.
    <https://mitpress.mit.edu/9780262631853/an-introduction-to-genetic-algorithms/>`_
    MIT Press.

    Levine, D., 1997.
    `Commentary—Genetic algorithms: A practitioner's view.
    <https://pubsonline.informs.org/doi/10.1287/ijoc.9.3.256>`_
    INFORMS Journal on Computing, 9(3), pp.256-259.

    Goldberg, D.E., 1994.
    `Genetic and evolutionary algorithms come of age.
    <https://dl.acm.org/doi/10.1145/175247.175259>`_
    Communications of the ACM, 37(3), pp.113-120.

    De Jong, K.A., 1993.
    `Are genetic algorithms function optimizer?.
    <https://www.sciencedirect.com/science/article/pii/B9780080948324500064>`_
    Foundations of Genetic Algorithms, pp.5-17.

    Forrest, S., 1993.
    `Genetic algorithms: Principles of natural selection applied to computation.
    <https://www.science.org/doi/10.1126/science.8346439>`_
    Science, 261(5123), pp.872-878.

    Mitchell, M., Holland, J. and Forrest, S., 1993.
    `When will a genetic algorithm outperform hill climbing.
    <https://proceedings.neurips.cc/paper/1993/hash/ab88b15733f543179858600245108dd8-Abstract.html>`_
    Advances in Neural Information Processing Systems (pp. 51-58).

    Holland, J.H., 1992.
    `Adaptation in natural and artificial systems: An introductory analysis with applications to
    biology, control, and artificial intelligence.
    <https://direct.mit.edu/books/book/2574/Adaptation-in-Natural-and-Artificial-SystemsAn>`_
    MIT press.
    
    Holland, J.H., 1992.
    `Genetic algorithms.
    <https://www.scientificamerican.com/article/genetic-algorithms/>`_
    Scientific American, 267(1), pp.66-73.

    Goldberg, D.E., 1989.
    `Genetic algorithms in search, optimization and machine learning.
    <https://www.goodreads.com/en/book/show/142613>`_
    Reading: Addison-Wesley.
 
    Goldberg, D.E. and Holland, J.H., 1988.
    `Genetic algorithms and machine learning.
    <https://link.springer.com/article/10.1023/A:1022602019183>`_
    Machine Learning, 3(2), pp.95-99.

    Holland, J.H., 1973.
    `Genetic algorithms and the optimal allocation of trials.
    <https://epubs.siam.org/doi/10.1137/0202009>`_    
    SIAM Journal on Computing, 2(2), pp.88-105.

    Holland, J.H., 1962.
    `Outline for a logical theory of adaptive systems.
    <https://dl.acm.org/doi/10.1145/321127.321128>`_
    Journal of the ACM, 9(3), pp.297-314.
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

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
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
