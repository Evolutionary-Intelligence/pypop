from optimizers.pso.opso import OPSO
from optimizers.pso.psogt import global_topology


class OPSOGT(OPSO):
    """Online Particle Swarm Optimizer with Global Topology (OPSOGT).

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
        OPSO.__init__(self, problems, options)
        self.topology = global_topology
