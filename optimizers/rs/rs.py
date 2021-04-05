from optimizers.core.optimizer import Optimizer


class RS(Optimizer):
    """Random (Stochastic) Search (RS).

    Reference
    ---------
    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        verbose_options = options.get('verbose_options')
        if verbose_options is None:
            self.verbose_options['frequency_verbose'] = 1000
        elif verbose_options.get('frequency_verbose') is None:
            self.verbose_options['frequency_verbose'] = 1000
        self.x = options.get('x')  # starting search point

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        raise NotImplementedError

    def _print_verbose_info(self):
        if self.verbose_options['verbose']:
            if not self.n_function_evaluations % self.verbose_options['frequency_verbose']:
                info = '  * Evaluations {:d}: best_so_far_y {:7.5e}'
                print(info.format(self.n_function_evaluations, self.best_so_far_y))
