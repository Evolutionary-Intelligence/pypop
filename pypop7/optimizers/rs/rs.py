import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class RS(Optimizer):
    """Random (stochastic) Search (optimization) (RS).

    This is the **base** (abstract) class for all `RS` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand. Recently, near all of its state-of-the-art versions adopt the
    **population-based** random sampling strategy for better exploration in the complex search space.

    .. note:: `"The topic of local search was reinvigorated in the early 1990s by surprisingly good results
       for large (combinatorial) problems ... and by the incorporation of randomness, multiple simultaneous searches,
       and other improvements."---[Russell&Norvig, 2022] <http://aima.cs.berkeley.edu/global-index.html>`_

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
              and with the following particular setting (`key`):
                * 'x' - initial (starting) point (`array_like`).

    Attributes
    ----------
    x     : `array_like`
            initial (starting) point.

    Methods
    -------

    References
    ----------
    Gao, K. and Sener, O., 2022, June.
    Generalizing Gaussian smoothing for random search.
    In International Conference on Machine Learning (pp. 7077-7101). PMLR.
    https://proceedings.mlr.press/v162/gao22f.html

    Bergstra, J. and Bengio, Y., 2012.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research, 13(2).
    https://www.jmlr.org/papers/v13/bergstra12a.html

    Appel, M.J., Labarre, R. and Radulovic, D., 2004.
    On accelerated random search.
    SIAM Journal on Optimization, 14(3), pp.708-731.
    https://epubs.siam.org/doi/abs/10.1137/S105262340240063X

    Schmidhuber, J., Hochreiter, S. and Bengio, Y., 2001.
    Evaluating benchmark problems by random guessing.
    A Field Guide to Dynamical Recurrent Networks, pp.231-235.
    https://ml.jku.at/publications/older/ch9.pdf

    Rastrigin, L.A., 1986.
    Random search as a method for optimization and adaptation.
    In Stochastic Optimization.
    https://link.springer.com/chapter/10.1007/BFb0007129

    Solis, F.J. and Wets, R.J.B., 1981.
    Minimization by random search techniques.
    Mathematics of Operations Research, 6(1), pp.19-30.
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
    https://tinyurl.com/25339c4x
    (*Since it was written originally in Russian, we cannot read it well. However, owing to its historical
    position, we still choose to include it here, which causes a nonstandard citation.*)

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
        # reset its default value from 10 to 1000, since typically more iterations can be run
        # for individual-based iterative search
        self.verbose = options.get('verbose', 1000)
        self._n_generations = 0  # number of generations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
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

    def _collect_results(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        fitness = Optimizer.optimize(self, fitness_function)
        x = self.initialize()
        y = self._evaluate_fitness(x, args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x = self.iterate()
            y = self._evaluate_fitness(x, args)
            self._n_generations += 1
        return self._collect_results(fitness, y)
