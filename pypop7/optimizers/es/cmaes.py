import numpy as np

from pypop7.optimizers.es.es import ES


class CMAES(ES):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Hansen, N., 2016.
    The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
    https://arxiv.org/abs/1604.00772

    https://github.com/CyberAgentAILab/cmaes
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        w_apostrophe = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)
        self._mu_eff = np.power(np.sum(w_apostrophe[:self.n_parents]), 2)/np.sum(
            np.power(w_apostrophe[:self.n_parents], 2))
        self._mu_eff_minus = np.power(np.sum(w_apostrophe[self.n_parents:]), 2) / np.sum(
            np.power(w_apostrophe[self.n_parents:], 2))
        self._alpha_cov = 2.0
        self.c_1 = self._alpha_cov/(np.power(self.ndim_problem + 1.3, 2) + self._mu_eff)
        self.c_w = self._set_c_w()
        self.c_s = (self._mu_eff + 2.0)/(self._mu_eff + self.ndim_problem + 5.0)
        self.d_sigma = self._set_d_sigma()
        self.c_c = self._set_c_c()

    def _set_c_w(self):
        # minus 1e-8 for large population size (according to https://github.com/CyberAgentAILab/cmaes)
        return np.minimum(1.0 - self.c_1 - 1e-8, self._alpha_cov*(self._mu_eff + 1.0/self._mu_eff - 2.0) /
                          (np.power(self.ndim_problem + 2.0, 2) + self._alpha_cov*self._mu_eff/2.0))

    def _set_d_sigma(self):
        return 1.0 + self.c_s + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0)

    def _set_c_c(self):
        return (4 + self._mu_eff/self.ndim_problem)/(self.ndim_problem + 4.0 + 2.0*self._mu_eff/self.ndim_problem)
