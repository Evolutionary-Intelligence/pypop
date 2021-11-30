from optimizers.es.maes import MAES


class FMAES(MAES):
    """Fast Matrix Adaptation Evolution Strategy (FMAES, Fast-(μ/μ_w, λ)-MA-ES).

    Reference
    ---------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700).
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    Large scale black-box optimization by limited-memory matrix adaptation.
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.
    https://ieeexplore.ieee.org/abstract/document/8410043

    Beyer, H.G. and Sendhoff, B., 2017.
    Simplify your covariance matrix adaptation evolution strategy.
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.
    https://ieeexplore.ieee.org/document/7875115

    https://homepages.fhv.at/hgb/downloads/ForDistributionFastMAES.tar    (see the official Matlab version)
    """
    def __init__(self, problem, options):
        options['_fast_version'] = True  # mandatory setting for FMAES
        MAES.__init__(self, problem, options)
