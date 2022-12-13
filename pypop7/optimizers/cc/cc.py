import numpy as np

from pypop7.optimizers.core.optimizer import Optimizer


class CC(Optimizer):
    """Cooperative Coevolution (CC).

    This is the **base** (abstract) class for all `CC` classes. Please use any of its instantiated subclasses to
    optimize the black-box problem at hand.

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
                * 'n_individuals' - number of individuals/samples, aka population size (`int`, default: `100`).

    References
    ----------
    Gomez, F., Schmidhuber, J. and Miikkulainen, R., 2008.
    Accelerated neural evolution through cooperatively coevolved synapses.
    Journal of Machine Learning Research, 9(31), pp.937-965.
    https://www.jmlr.org/papers/v9/gomez08a.html

    Panait, L., Tuyls, K. and Luke, S., 2008.
    Theoretical advantages of lenient learners: An evolutionary game theoretic perspective.
    Journal of Machine Learning Research, 9, pp.423-457.
    https://jmlr.org/papers/volume9/panait08a/panait08a.pdf

    Schmidhuber, J., Wierstra, D., Gagliolo, M. and Gomez, F., 2007.
    Training recurrent networks by evolino.
    Neural Computation, 19(3), pp.757-779.
    https://direct.mit.edu/neco/article-abstract/19/3/757/7156/Training-Recurrent-Networks-by-Evolino

    Gomez, F.J. and Schmidhuber, J., 2005, June.
    Co-evolving recurrent neurons learn deep memory POMDPs.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation (pp. 491-498).
    https://dl.acm.org/doi/10.1145/1068009.1068092

    Fan, J., Lau, R. and Miikkulainen, R., 2003.
    Utilizing domain knowledge in neuroevolution.
    In International Conference on Machine Learning (pp. 170-177).
    https://www.aaai.org/Library/ICML/2003/icml03-025.php

    Potter, M.A. and De Jong, K.A., 2000.
    Cooperative coevolution: An architecture for evolving coadapted subcomponents.
    Evolutionary Computation, 8(1), pp.1-29.
    https://direct.mit.edu/evco/article/8/1/1/859/Cooperative-Coevolution-An-Architecture-for

    Gomez, F.J. and Miikkulainen, R., 1999, July.
    Solving non-Markovian control tasks with neuroevolution.
    In Proceedings of International Joint Conference on Artificial Intelligence (pp. 1356-1361).
    https://www.ijcai.org/Proceedings/99-2/Papers/097.pdf

    Moriarty, D.E. and Mikkulainen, R., 1996.
    Efficient reinforcement learning through symbiotic evolution.
    Machine Learning, 22(1), pp.11-32.
    https://link.springer.com/article/10.1023/A:1018004120707

    Moriarty, D.E. and Miikkulainen, R., 1995.
    Efficient learning from delayed rewards through symbiotic evolution.
    In International Conference on Machine Learning (pp. 396-404). Morgan Kaufmann.
    https://www.sciencedirect.com/science/article/pii/B9781558603776500566

    Potter, M.A. and De Jong, K.A., 1994, October.
    A cooperative coevolutionary approach to function optimization.
    In International Conference on Parallel Problem Solving from Nature (pp. 249-257).
    Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-58484-6_269
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.n_individuals = options.get('n_individuals', 100)  # number of individuals/samples, aka population size
        self._n_generations = 0  # initial number of generations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            fitness.extend(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
