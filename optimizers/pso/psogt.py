import numpy as np

from optimizers.pso.pso import PSO


# helper function
def global_topology(p_x, p_y, i):
    return p_x[np.argmin(p_y)], i


class PSOGT(PSO):
    """Particle Swarm Optimizer with Global Topology (PSOGT).

    Reference
    ---------
    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146
    """
    def __init__(self, problems, options):
        PSO.__init__(self, problems, options)
        self.topology = global_topology
