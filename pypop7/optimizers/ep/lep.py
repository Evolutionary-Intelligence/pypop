import numpy as np
from scipy.stats import levy_stable

from pypop7.optimizers.ep.cep import CEP


class LEP(CEP):
    """Lévy distribution based Evolutionary Programming (LEP).

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
                * 'sigma'          - initial global step-size, aka mutation strength (`float`),
                * 'n_individuals'  - number of offspring, aka offspring population size (`int`, default: `100`),
                * 'q'              - number of opponents for pairwise comparisons (`int`, default: `10`),
                * 'tau'            - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*np.sqrt(self.ndim_problem))`),
                * 'tau_apostrophe' - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*self.ndim_problem)`.

    Examples
    --------
    Use the Evolutionary Programming optimizer `LEP` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ep.lep import LEP
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5 * numpy.ones((2,)),
       ...            'upper_boundary': 5 * numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.1}
       >>> lep = LEP(problem, options)  # initialize the optimizer class
       >>> results = lep.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"LEP: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       LEP: 5000, 0.0359694938656471

    For its correctness checking, refer to `this code-based repeatability report
    <https://tinyurl.com/2dc6ym6j>`_ for more details.

    Attributes
    ----------
    n_individuals  : `int`
                     number of offspring, aka offspring population size.
    q              : `int`
                     number of opponents for pairwise comparisons.
    sigma          : `float`
                     initial global step-size, aka mutation strength.
    tau            : `float`
                     learning rate of individual step-sizes self-adaptation.
    tau_apostrophe : `float`
                     learning rate of individual step-sizes self-adaptation.

    References
    ----------
    Lee, C.Y. and Yao, X., 2004.
    Evolutionary programming using mutations based on the Lévy probability distribution.
    IEEE Transactions on Evolutionary Computation, 8(1), pp.1-13.
    https://ieeexplore.ieee.org/document/1266370
    """
    def __init__(self, problem, options):
        CEP.__init__(self, problem, options)

    def iterate(self, x=None, sigmas=None, y=None, xx=None, ss=None, yy=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y, xx, ss, yy
            ss[i] = sigmas[i]*np.exp(self.tau_apostrophe*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)) + self.tau*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)))
            xx[i] = x[i] + ss[i]*levy_stable.rvs(alpha=1.8, beta=1, size=(self.ndim_problem,),
                                                 random_state=self.rng_optimization)
            yy[i] = self._evaluate_fitness(xx[i], args)
        new_x = np.vstack((xx, x))
        new_sigmas = np.vstack((ss, sigmas))
        new_y = np.hstack((yy, y))
        n_win = np.zeros((2*self.n_individuals,))  # number of win
        for i in range(2*self.n_individuals):
            for j in self.rng_optimization.choice(np.setdiff1d(range(2*self.n_individuals), i),
                                                  size=self.q, replace=False):
                if new_y[i] < new_y[j]:
                    n_win[i] += 1
        order = np.argsort(-n_win)[:self.n_individuals]
        x[:self.n_individuals] = new_x[order]
        sigmas[:self.n_individuals] = new_sigmas[order]
        y[:self.n_individuals] = new_y[order]
        self._n_generations += 1
        return x, sigmas, y, xx, ss, yy
