import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class RS(Optimizer):
    """Random (stochastic) Search (optimization) (RS).

    This is the **base** (abstract) class for all RS classes (also including **local search**, particularly
    its *randomized* versions). At least its two methods (`initialize`, `iterate`) should be implemented by
    any of its subclasses.

    .. note:: `"The topic of local search was reinvigorated in the early 1990s by surprisingly good results
       for large ... problems ... and by the incorporation of randomness, multiple simultaneous searches,
       and other improvements." <http://aima.cs.berkeley.edu/global-index.html>`_

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
                * 'record_fitness'           - flag to record fitness list to output results (`bool`, default: `False`),
                * 'record_fitness_frequency' - function evaluations frequency of recording (`int`, default: `1000`),

                  * if `record_fitness` is set to `False`, it will be ignored,
                  * if `record_fitness` is set to `True` and it is set to 1, all fitness generated during optimization
                    will be saved into output results.

                * 'verbose'                  - flag to print verbose info during optimization (`bool`, default: `True`),
                * 'verbose_frequency'        - frequency of printing verbose info (`int`, default: `1000`);
              and with one particular setting (`key`):
                * 'x'     - initial (starting) point (`array_like`).

    Attributes
    ----------
    x     : `array_like`
            initial (starting) point.

    Methods
    -------

    References
    ----------
    Bergstra, J. and Bengio, Y., 2012.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research, 13(2).
    https://www.jmlr.org/papers/v13/bergstra12a.html

    Appel, M.J., Labarre, R. and Radulovic, D., 2004.
    On accelerated random search.
    SIAM Journal on Optimization, 14(3), pp.708-731.
    https://epubs.siam.org/doi/abs/10.1137/S105262340240063X

    Solis, F.J. and Wets, R.J.B., 1981.
    Minimization by random search techniques.
    Mathematics of operations research, 6(1), pp.19-30.
    https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.19

    Schrack, G. and Choit, M., 1976.
    Optimized relative step size random searches.
    Mathematical Programming, 10(1), pp.230-244.
    https://link.springer.com/article/10.1007/BF01580669

    Schumer, M.A. and Steiglitz, K., 1968.
    Adaptive step size random search.
    IEEE Transactions on Automatic Control, 13(3), pp.270-276.
    https://ieeexplore.ieee.org/abstract/document/1098903

    Matyas, J., 1965.
    Random optimization.
    Automation and Remote control, 26(2), pp.246-253.
    https://tinyurl.com/25339c4x  (in Russian)

    Rastrigin, L.A., 1963.
    The convergence of the random search method in the extremal control of a many parameter system.
    Automaton & Remote Control, 24, pp.1337-1342.
    https://tinyurl.com/djfdnpx4

    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.x = options.get('x')  # initial (starting) point
        # reset its default value from 10 to 1000 (since typically more iterations are run)
        if options.get('verbose_frequency') is None:
            self.verbose_frequency = 1000
        self._n_generations = 0  # number of generations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        fitness = Optimizer.optimize(self, fitness_function)
        is_initialization = True
        while not self._check_terminations():
            if is_initialization:
                x = self.initialize()
                is_initialization = False
            else:
                x = self.iterate()
            y = self._evaluate_fitness(x, args)
            if self.record_fitness:
                fitness.append(y)
            self._print_verbose_info(y)
            self._n_generations += 1
        return self._collect_results(fitness)
