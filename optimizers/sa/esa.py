import time
import numpy as np

from optimizers.sa.sa import SA


class ESA(SA):
    """Enhanced Simulated Annealing (ESA).

    Note that current implementation is slightly different from the original paper,
    with the following code improvements:
        1. simplify Space Partitioning (Step 2):
            Each dimension can be chosen only once for one movement cycle,
        2. modify Temperature Adjustment (Step 6):
            Temperature should be set properly via hyper-parameter optimization,
        3. delete Four Nonexclusive Stopping Tests (Step 8):
            We use the default stopping conditions suggested by the pypop library.

    Reference
    ---------
    Siarry, P., Berthiau, G., Durdin, F. and Haussy, J., 1997.
    Enhanced simulated annealing for globally minimizing functions of many-continuous variables.
    ACM Transactions on Mathematical Software, 23(2), pp.209-228.
    https://dl.acm.org/doi/abs/10.1145/264029.264043
    """
    def __init__(self, problem, options):
        SA.__init__(self, problem, options)
        # for Step 5: Test for End of Temperature Stage
        self.N1 = options.get('N1', 12)  # for accepted moves
        self.N2 = options.get('N2', 100)  # for attempted moves
        # for Step 7: Step Vector Adjustment
        self.RATMAX = options.get('RATMAX', 0.2)  # condition to extend step vector
        self.EXTSTP = options.get('EXTSTP', 2)  # extending factor
        self.RATMIN = options.get('RATMIN', 0.05)  # condition to shrink step vector
        self.SHRSTP = options.get('SHRSTP', 0.5)  # shrinking factor
        # system parameters at current temperature stage
        self.MVOKST = 0  # number of accepted moves
        self.MOKST = np.zeros((self.ndim_problem,))  # numbers of accepted moves for each dimension
        self.NMVST = 0  # number of attempted moves
        self.MTOTST = np.zeros((self.ndim_problem,))  # numbers of attempted moves for each dimension

    def perform_cycle(self, args=None):  # Step 2, 3, 4
        # Step 2: Space Partitioning
        #  Note that the original version seems to be excessively complex.
        #  Here we only consider one dimension for each movement and make sure that all
        #  dimensions are optimized in one movement cycle, though their order is randomly generated.
        p = self.rng_optimization.permutation(self.ndim_problem)  # len(p) == n
        fitness = []
        for k in p:
            if self._check_terminations():
                break
            # Step 3: Execution of One Movement
            #   Here we execute one movement for each dimension (a very simplified setting).
            x = np.copy(self.parent_x)
            x[k] += self.rng_optimization.uniform(-1, 1) * self.v[k]
            y = self._evaluate_fitness(x, args)
            if self.record_options['record_fitness']:
                fitness.append(y)
            self._print_verbose_info()
            # Step 4: Acceptance or Rejection of the Movement
            diff = self.parent_y - y
            self.MTOTST[k] += 1
            self.NMVST += 1
            if (diff >= 0) or (self.rng_optimization.random() < np.exp(diff / self.T)):
                self.parent_x, self.parent_y = x, y
                self.MOKST[k] += 1
                self.MVOKST += 1
        return fitness

    def adjust_step_vector(self):  # Step 7: Step Vector Adjustment
        for k in range(self.ndim_problem):
            rok = self.MOKST[k] / self.MTOTST[k]
            if rok > self.RATMAX:
                self.v[k] *= self.EXTSTP
            elif rok < self.RATMIN:
                self.v[k] *= self.SHRSTP

    def reset_parameters(self):  # Step 9: Initialization of a New Temperature Stage
        self.MVOKST = 0
        self.MOKST = np.zeros((self.ndim_problem,))
        self.NMVST = 0
        self.MTOTST = np.zeros((self.ndim_problem,))

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        # Step 1: Initializations
        fitness = self.initialize(args)  # store all fitness generated during search
        while not self._check_terminations():
            # Step 5: Test for End of Temperature Stage
            while (self.MVOKST < self.N1 * self.ndim_problem) and (self.NMVST < self.N2 * self.ndim_problem):
                # Step 2, 3, and 4 are combined into `self.perform_cycle`:
                fitness.extend(self.perform_cycle(args))
                if self._check_terminations():
                    break
            # Step 6: Temperature Adjustment
            #  Here we don't execute the original version, which is complicated (fitness-dependent).
            #  Instead, we execute a very simple version from [Corana et al., 1987] and expect that
            #  in practice such a hyper-parameter will be properly set by systematic search.
            self.T *= self.r_T
            # Step 7: Step Vector Adjustment
            self.adjust_step_vector()
            # Step 8: Four Nonexclusive Stopping Tests
            #  Here we don't execute them, since their settings are problem-dependent.
            #  In practice, it appears to be difficult to provide problem-independent executions.
            # Step 9: Initialization of a New Temperature Stage
            self.reset_parameters()
        if self.record_options['record_fitness']:
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return self._collect_results()
