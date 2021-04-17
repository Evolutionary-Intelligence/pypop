from optimizers.pso.pso import PSO


# helper function
def ring_topology(p_x, p_y, i):
    if len(p_y) < 3:
        raise ValueError('Swarm size should (at least) >= 3.')
    if i == 0:
        left, right = len(p_y) - 1, 1
    elif i == len(p_y) - 1:
        left, right = len(p_y) - 2, 0
    else:
        left, right = i - 1, i + 1
    better = p_x[left] if p_y[left] <= p_y[right] else p_x[right]
    return better, i


class PSORT(PSO):
    """Particle Swarm Optimizer with Ring Topology (PSORT).

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
        self.topology = ring_topology
