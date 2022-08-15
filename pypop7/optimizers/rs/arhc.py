import numpy as np

from pypop7.optimizers.rs.rhc import RHC


class ARHC(RHC):
    """Annealed Random Hill Climber (ARHC).

    .. note:: The search performance of `ARHC` depends **heavily** on the setting of *temperature* of the
       annealing process. However, its setting is a non-trival task, since it may vary among different
       problems and even among different optimization stages for the given problem at hand.

       It is **highly recommended** to first attempt other more advanced methods for LSBBO.

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
              and with three particular settings (`keys`):
                * 'x'                         - initial (starting) point (`array_like`),
                * 'sigma'                     - initial (global) step-size (`float`),
                * 'temperature'               - annealing temperature (`float`),
                * initialization_distribution - random sampling distribution for starting point initialization (
                  `int`, default: `1`).

                  * `1`: *uniform* distribution is used for random sampling,
                  * `0`: *standard normal* distribution is used for random sampling.

    Examples
    --------
    Use the Random Search optimizer `ARHC` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.rs.arhc import ARHC
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'x': 3 * numpy.ones((2,)),
       ...            'sigma': 0.1,
       ...            'temperature': 1.5}
       >>> arhc = ARHC(problem, options)  # initialize the optimizer class
       >>> results = arhc.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"Annealed-RHC: {results['n_function_evaluations']}, {results['best_so_far_y']}")
         * Generation 0: best_so_far_y 3.60400e+03, min(y) 3.60400e+03 & Evaluations 1
         * Generation 1000: best_so_far_y 2.64114e-04, min(y) 5.91452e-01 & Evaluations 1001
         * Generation 2000: best_so_far_y 2.64114e-04, min(y) 1.04657e+01 & Evaluations 2001
         * Generation 3000: best_so_far_y 2.64114e-04, min(y) 4.86862e-01 & Evaluations 3001
         * Generation 4000: best_so_far_y 2.64114e-04, min(y) 6.62829e-02 & Evaluations 4001
       Annealed-RHC: 5000, 0.0002641143073543329

    Attributes
    ----------
    x                           : `array_like`
                                  initial (starting) point.
    sigma                       : `float`
                                  (global) step-size.
    temperature                 : `float`
                                  annealing temperature.
    initialization_distribution : `int`
                                  random sampling distribution for initialization of starting point.

    References
    ----------
    https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/hillclimber.py
    """
    def __init__(self, problem, options):
        RHC.__init__(self, problem, options)
        self.temperature = options.get('temperature')
        if self.temperature is None:
            raise ValueError('`temperature` should be set.')
        self._parent_x = np.copy(self.best_so_far_x)
        self._parent_y = np.copy(self.best_so_far_y)

    def iterate(self):  # draw a sample via mutating the parent individual
        noise = self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        return self._parent_x + self.sigma*noise

    def _evaluate_fitness(self, x, args=None):
        y = RHC._evaluate_fitness(self, x, args)
        # update parent solution and fitness
        diff = y - self._parent_y
        if (diff < 0) or (self.rng_optimization.random() < np.exp(-diff / self.temperature)):
            self._parent_x, self._parent_y = np.copy(x), y
        return y
