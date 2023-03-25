from pypop7.optimizers.core.optimizer import Optimizer


class BO(Optimizer):
    """Bayesian Optimization (BO).

    References
    ----------
    https://bayesoptbook.com/

    https://bayesopt-tutorial.github.io/
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)

    def initialize(self):
        raise NotImplementedError

    def iterate(self):  # for each iteration (generation)
        raise NotImplementedError
