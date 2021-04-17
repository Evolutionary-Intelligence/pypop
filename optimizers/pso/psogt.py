import numpy as np

from optimizers.pso.pso import PSO


# helper function
def global_topology(p_x, p_y, i):
    return p_x[np.argmin(p_y)], i


class PSOGT(PSO):
    """Particle Swarm Optimizer with Global Topology (PSOGT).

    Reference
    ---------
    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (Vol. 4, pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968

    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146
    """
    def __init__(self, problems, options):
        PSO.__init__(self, problems, options)
        self.topology = global_topology
