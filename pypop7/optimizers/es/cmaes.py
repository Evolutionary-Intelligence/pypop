import numpy as np  # engine for numerical computing

from pypop7.optimizers.es.es import ES  # abstract class of all Evolution Strategies (ES) classes


class CMAES(ES):
    """Covariance Matrix Adaptation Evolution Strategy (CMAES).

    .. note:: `CMAES` is widely recognized as one of **State-Of-The-Art (SOTA)** evolutionary algorithms for continuous
       black-box optimization (BBO), according to one well-recognized `Nature <https://doi.org/10.1038/nature14544>`_
       review of Evolutionary Computation (EC).

       For some (rather all) applications of `CMAES`, please refer to e.g., `[Wang et al., 2024, ICLR]
       <https://arxiv.org/abs/2404.00451>`_, `[Gil-Fuster et al., 2024, Nature Communications]
       <https://www.nature.com/articles/s41467-024-45882-z>`_, `[Jin et al., 2024]
       <https://link.springer.com/article/10.1007/s11044-024-09982-4>`_, `[Koginov et al., 2024, TMRB]
       <https://ieeexplore.ieee.org/document/10302449>`_, `[Elfikky et al., 2024, LWC]
       <https://ieeexplore.ieee.org/abstract/document/10531788>`_, `[Hooper et al., 2024, RSIF]
       <https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0141>`_, `[Yuan et al., 2024, MNRAS]
       <https://academic.oup.com/mnras/article/530/1/947/7643636#>`_, `[Bruel et al., 2024]
       <https://www.biorxiv.org/content/10.1101/2024.04.12.589164v1>`_, `[Li et al., 2024]
       <https://arxiv.org/abs/2403.17009>`_, `[Liu et al., 2024]
       <https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16962>`_, `[Martin, 2024, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37378922>`_, `[Milekovic et al., 2023, Nature Medicine]
       <https://doi.org/10.1038/s41591-023-02584-1>`_, `[Falk et al., 2023, PNAS]
       <https://www.pnas.org/doi/abs/10.1073/pnas.2219558120>`_, `[Thamm&Rosenow, 2023, PRL]
       <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.116202>`_, `[Brea et al., 2023, Nature
       Communications] <https://www.nature.com/articles/s41467-023-38570-x>`_, `[Slade et al., 2022, Nature]
       <https://www.nature.com/articles/s41586-022-05191-1>`_, `[Croon et al., 2022, Nature]
       <https://www.nature.com/articles/s41586-022-05182-2>`_, `[Rudolph et al., 2022, Nature Communications]
       <https://www.nature.com/articles/s41467-023-43908-6>`_, `[Cazenille et al., 2022, Bioinspiration & Biomimetics]
       <https://iopscience.iop.org/article/10.1088/1748-3190/ac7fd1>`_, `[Franks et al., 2021]
       <https://www.biorxiv.org/content/10.1101/2021.09.13.460170v1.abstract>`_, `[Yuan et al., 2021, MNRAS]
       <https://academic.oup.com/mnras/article/502/3/3582/6122578>`_, `[Löffler et al., 2021, Nature Communications]
       <https://www.nature.com/articles/s41467-021-22017-2>`_, `[Papadopoulou et al., 2021, JPCB]
       <https://pubs.acs.org/doi/10.1021/acs.jpcb.1c07562>`_, `[Barkley, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37368472>`_, `[Fernandes, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37370084>`_, `[Quinlivan, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37369463>`_, `[Vasios et al., 2020, Soft Robotics]
       <https://www.liebertpub.com/doi/full/10.1089/soro.2018.0149>`_, `[Pal et al., 2020]
       <https://iopscience.iop.org/article/10.1088/1361-665X/abbd1d>`_, `[Lei, 2020, Ph.D. Dissertation (University of Oxford)]
       <https://tinyurl.com/yzkjwr34>`_, `[Yang et al., 2019, Journal of Aircraft]
       <https://arc.aiaa.org/doi/full/10.2514/1.C034873>`_, `[Ong et al., 2019, PLOS Computational Biology]
       <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006993>`_, `[Zhang et al., 2017, Science]
       <https://www.science.org/doi/full/10.1126/science.aal5054>`_, `[Loshchilov&Hutter, 2016]
       <https://arxiv.org/abs/1604.07269>`_, `[Molinari et al., 2014, AIAAJ]
       <https://arc.aiaa.org/doi/full/10.2514/1.J052715>`_, `[Melton, 2014, Acta Astronautica]
       <https://www.sciencedirect.com/science/article/pii/S0094576514002318>`_, `[Khaira et al., 2014, ACS Macro Lett.]
       <https://pubs.acs.org/doi/full/10.1021/mz5002349>`_, `[Wang et al., 2010, TOG]
       <https://dl.acm.org/doi/10.1145/1778765.1778810>`_, `[Wampler&Popović, 2009, TOG]
       <https://dl.acm.org/doi/10.1145/1531326.1531366>`_
       `RoboCup <https://doi.org/10.1007/s10458-024-09642-z>`_, 2014 3D Simulation League Competition Champions,
       `[Muller et al., 2001, AIAAJ] <https://arc.aiaa.org/doi/abs/10.2514/2.1342>`_,
       to name a few.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the black-box optimizer `CMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.cmaes import CMAES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 3.0}  # global step-size may need to be fine-tuned for better performance
       >>> cmaes = CMAES(problem, options)  # to initialize the optimizer class
       >>> results = cmaes.optimize()  # to run the optimization/evolution process
       >>> print(f"CMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CMAES: 5000, 0.0017

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_cmaes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_cmaes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_cmaes.py>`_.

    Attributes
    ----------
    best_so_far_x : `array_like`
                    final best-so-far solution found during entire optimization.
    best_so_far_y : `array_like`
                    final best-so-far fitness found during entire optimization.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size / sample size.
    n_parents     : `int`
                    number of parents, aka parental population size / number of positively selected search points.
    sigma         : `float`
                    final global step-size, aka mutation strength (updated during optimization).

    References
    ----------
    https://cma-es.github.io/

    `Hansen, N. <http://www.cmap.polytechnique.fr/~nikolaus.hansen/>`_, 2023.
    `The CMA evolution strategy: A tutorial.
    <https://arxiv.org/abs/1604.00772>`_
    arXiv preprint arXiv:1604.00772.

    Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017.
    `Information-geometric optimization algorithms: A unifying picture via invariance principles.
    <https://jmlr.org/papers/v18/14-467.html>`_
    Journal of Machine Learning Research, 18(18), pp.1-65.

    Hansen, N., Atamna, A. and Auger, A., 2014, September.
    `How to assess step-size adaptation mechanisms in randomised search.
    <https://link.springer.com/chapter/10.1007/978-3-319-10762-2_6>`_
    In International Conference on Parallel Problem Solving From Nature (pp. 60-69). Springer, Cham.

    Kern, S., Müller, S.D., Hansen, N., Büche, D., Ocenasek, J. and Koumoutsakos, P., 2004.
    `Learning probability distributions in continuous evolutionary algorithms–a comparative review.
    <https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e>`_
    Natural Computing, 3, pp.77-112.

    Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003.
    `Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES).
    <https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized>`_
    Evolutionary Computation, 11(1), pp.1-18.

    Hansen, N. and Ostermeier, A., 2001.
    `Completely derandomized self-adaptation in evolution strategies.
    <https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in>`_
    Evolutionary Computation, 9(2), pp.159-195.

    Hansen, N. and Ostermeier, A., 1996, May.
    `Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation.
    <https://ieeexplore.ieee.org/abstract/document/542381>`_
    In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.

    Please refer to its *lightweight* Python implementation from `cyberagent.ai
    <https://cyberagent.ai/>`_:
    https://github.com/CyberAgentAILab/cmaes

    Please refer to its *official* Python implementation from `Hansen, N.
    <http://www.cmap.polytechnique.fr/~nikolaus.hansen/>`_:
    https://github.com/CMA-ES/pycma
    """
    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options)
        assert self.n_individuals >= 2
        self._w, self._mu_eff, self._mu_eff_minus = None, None, None  # variance effective selection mass
        # c_s (c_σ) -> decay rate for the cumulating path for the step-size control
        self.c_s, self.d_sigma = None, None  # for cumulative step-length adaptation (CSA)
        self._p_s_1, self._p_s_2 = None, None  # for evolution path update of CSA
        self._p_c_1, self._p_c_2 = None, None  # for evolution path update of CMA
        # c_c -> decay rate for cumulating path for the rank-one update of CMA
        # c_1 -> learning rate for the rank-one update of CMA
        # c_w (c_μ) -> learning rate for the rank-µ update of CMA
        self.c_c, self.c_1, self.c_w, self._alpha_cov = None, None, None, 2.0  # for CMA (c_w -> c_μ)
        self._save_eig = options.get('_save_eig', False)  # whether or not save eigenvalues and eigenvectors

    def _set_c_c(self):
        """Set decay rate of evolution path for the rank-one update of CMA.
        """
        return (4.0 + self._mu_eff / self.ndim_problem) / (
                self.ndim_problem + 4.0 + 2.0 * self._mu_eff / self.ndim_problem)

    def _set_c_w(self):
        return np.minimum(1.0 - self.c_1, self._alpha_cov*(1.0/4.0 + self._mu_eff + 1.0/self._mu_eff - 2.0) /
                          (np.square(self.ndim_problem + 2.0) + self._alpha_cov*self._mu_eff/2.0))

    def _set_d_sigma(self):
        return 1.0 + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0) + self.c_s

    def initialize(self, is_restart=False):
        w_a = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)  # w_apostrophe
        self._mu_eff = np.square(np.sum(w_a[:self.n_parents]))/np.sum(np.square(w_a[:self.n_parents]))
        self._mu_eff_minus = np.square(np.sum(w_a[self.n_parents:]))/np.sum(np.square(w_a[self.n_parents:]))
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0)/(self.ndim_problem + self._mu_eff + 5.0))
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self.c_c = self.options.get('c_c', self._set_c_c())
        self.c_1 = self.options.get('c_1', self._alpha_cov/(np.square(self.ndim_problem + 1.3) + self._mu_eff))
        self.c_w = self.options.get('c_w', self._set_c_w())
        w_min = np.min([1.0 + self.c_1/self.c_w, 1.0 + 2.0*self._mu_eff_minus/(self._mu_eff + 2.0),
                        (1.0 - self.c_1 - self.c_w)/(self.ndim_problem*self.c_w)])
        self._w = np.where(w_a >= 0, 1.0/np.sum(w_a[w_a > 0])*w_a, w_min/(-np.sum(w_a[w_a < 0]))*w_a)
        self._p_s_1, self._p_s_2 = 1.0 - self.c_s, np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)
        self._p_c_1, self._p_c_2 = 1.0 - self.c_c, np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)
        x = np.empty((self.n_individuals, self.ndim_problem))  # a population of search points (individuals, offspring)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_s = np.zeros((self.ndim_problem,))  # evolution path (p_σ) for cumulative step-length adaptation (CSA)
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation (CMA)
        cm = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        e_ve = np.eye(self.ndim_problem)  # eigenvectors of `cm` (orthogonal matrix)
        e_va = np.ones((self.ndim_problem,))  # square roots of eigenvalues of `cm` (in diagonal rather matrix form)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        d = np.empty((self.n_individuals, self.ndim_problem))
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def iterate(self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, y, d
            # produce a spherical (isotropic) Gaussian distribution (Nikolaus Hansen, 2023)
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Gaussian noise for mutation
            d[k] = np.dot(e_ve @ np.diag(e_va), z)
            x[k] = mean + self.sigma*d[k]  # offspring individual
            y[k] = self._evaluate_fitness(x[k], args)  # fitness
        return x, y, d

    def update_distribution(self, x=None, p_s=None, p_c=None, cm=None, e_ve=None, e_va=None, y=None, d=None):
        order = np.argsort(y)  # to rank all offspring individuals
        wd = np.dot(self._w[:self.n_parents], d[order[:self.n_parents]])
        # update distribution mean via weighted recombination
        mean = np.dot(self._w[:self.n_parents], x[order[:self.n_parents]])
        # update global step-size: cumulative path length control / cumulative step-size control /
        #   cumulative step length adaptation (CSA)
        cm_minus_half = e_ve @ np.diag(1.0/e_va) @ e_ve.T
        p_s = self._p_s_1*p_s + self._p_s_2*np.dot(cm_minus_half, wd)
        self.sigma *= np.exp(self.c_s/self.d_sigma*(np.linalg.norm(p_s)/self._e_chi - 1.0))
        # update covariance matrix (CMA)
        h_s = 1.0 if np.linalg.norm(p_s)/np.sqrt(1.0 - np.power(1.0 - self.c_s, 2*(self._n_generations + 1))) < (
                1.4 + 2.0/(self.ndim_problem + 1.0))*self._e_chi else 0.0
        p_c = self._p_c_1*p_c + h_s*self._p_c_2*wd
        w_o = self._w*np.where(self._w >= 0, 1.0, self.ndim_problem/(np.square(
            np.linalg.norm(cm_minus_half @ d.T, axis=0)) + 1e-8))
        cm = ((1.0 + self.c_1*(1.0 - h_s)*self.c_c*(2.0 - self.c_c) - self.c_1 - self.c_w*np.sum(self._w))*cm +
              self.c_1*np.outer(p_c, p_c))  # rank-one update
        for i in range(self.n_individuals):  # rank-μ update (to estimate variances of sampled *steps*)
            cm += self.c_w*w_o[i]*np.outer(d[order[i]], d[order[i]])
        # do eigen-decomposition and return both eigenvalues and eigenvectors
        cm = (cm + np.transpose(cm))/2.0  # to ensure symmetry of covariance matrix
        # return eigenvalues and eigenvectors of a symmetric matrix
        e_va, e_ve = np.linalg.eigh(cm)  # e_va -> eigenvalues, e_ve -> eigenvectors
        e_va = np.sqrt(np.where(e_va < 0.0, 1e-8, e_va))  # to avoid negative eigenvalues
        # e_va: squared root of eigenvalues -> interpreted as individual step-sizes and its diagonal entries are
        #       standard deviations of different components (from Nikolaus Hansen, 2023)
        cm = e_ve @ np.diag(np.square(e_va)) @ np.transpose(e_ve)  # to recover covariance matrix
        return mean, p_s, p_c, cm, e_ve, e_va

    def restart_reinitialize(self, x=None, mean=None, p_s=None, p_c=None,
                             cm=None, e_ve=None, e_va=None, y=None, d=None):
        if ES.restart_reinitialize(self, y):
            x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize(True)
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, y, d = self.iterate(x, mean, e_ve, e_va, y, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, p_s, p_c, cm, e_ve, e_va = self.update_distribution(x, p_s, p_c, cm, e_ve, e_va, y, d)
            if self.is_restart:
                x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.restart_reinitialize(
                    x, mean, p_s, p_c, cm, e_ve, e_va, y, d)
        results = self._collect(fitness, y, mean)
        results['p_s'] = p_s
        results['p_c'] = p_c
        # by default do *NOT* save eigenvalues and eigenvectors (with *quadratic* space complexity)
        if self._save_eig:
            results['e_va'], results['e_ve'] = e_va, e_ve
        return results
