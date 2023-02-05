import numpy as np

from pypop7.optimizers.es.es import ES


class VKDCMA(ES):
    """
    Akimoto, Y. and Hansen, N., 2016, September.
    Online model selection for restricted covariance matrix adaptation.
    In Parallel Problem Solving from Nature. Springer International Publishing.
    https://link.springer.com/chapter/10.1007/978-3-319-45823-6_1

    Akimoto, Y. and Hansen, N., 2016, July.
    Projection-based restricted covariance matrix adaptation for high dimension.
    In Proceedings of Annual Genetic and Evolutionary Computation Conference 2016 (pp. 197-204). ACM.
    https://dl.acm.org/doi/abs/10.1145/2908812.2908863
    """
