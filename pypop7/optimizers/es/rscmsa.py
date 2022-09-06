import numpy as np
from scipy.stats import norm

from pypop7.optimizers.es.es import ES


def calculate_mahalanobis_distance(y, x, inv_cov):
    if y.ndim == 1:
        return np.sqrt(np.dot(np.dot(y - x, inv_cov), (y - x)))
    if y.ndim == 2:
        return np.array([np.sqrt(np.dot(np.dot(yy - x, inv_cov), (yy - x))) for yy in y])


class RSCMSA(ES):
    """Covariance Matrix Self-Adaptation Evolution Strategy with Repelling Subpopulations (RS-CMSA)

    Reference
    ---------
    Beyer, H.G. and Sendhoff, B., 2008.
    Covariance matrix adaptation revisited–the CMSA evolution strategy.
    In International Conference on Parallel Problem-Solving from Nature (pp. 123-132). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-540-87700-4_13

    Ahrari, A., Deb, K. and Preuss, M., 2017.
    Multimodal optimization by covariance matrix self-adaptation evolution strategy with repelling subpopulations.
    Evolutionary computation, 25(3), pp.439-471.
    https://doi.org/10.1162/evco_a_00182
    """

    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.tau = np.sqrt(0.5/self.ndim_problem)  # the learning rate of global step size
        self.c_cov = self.n_parents/(self.n_parents + self.ndim_problem * (self.ndim_problem + 1))  # 1/tau_c
        self.n_elt = np.maximum(1, int(0.15 * self.n_individuals))  # number of elites
        self.tau_hat_d = np.sqrt(1/self.ndim_problem)  # the learning rate of the normalized taboo distance
        self.c_red = np.power(0.99, 1/self.ndim_problem)  # the size of each taboo region shrinks by (1-self.c_red)
        self.alpha_new = options.get('alpha_new', 0.5)  # target rate of basin identification
        self.c_threshold = options.get('c_threshold', 0.01)  # criticality threshold for taboo points
        self.n_s = options.get('n_subpopulations', 10)  # number of subpopulations
        self.p = options.get('p', 25)  # percentile
        self.budget = options.get('budget', 10)  # max evaluation budget for the hill-valley function
        self.hat_d_0 = options.get('hat_d_0', 1)  # default value of the normalized taboo distance
        self.n_condition = options.get('n_condition', 1e14)  # condition number
        self.sigma = options.get('sigma', 0.25)
        self._n_s_bak = np.copy(self.n_s)
        self._best_so_far_y_bak = np.copy(self.best_so_far_y)

    def initialize(self, args=None):
        archive_x, archive_f, archive_d = np.empty((0, self.ndim_problem)), np.empty((0,)), np.empty((0,))
        return archive_x, archive_f, archive_d

    def find_critical_points_descend_index(self, x, sigma, max_eig_sqrt, archive_x, archive_d):
        if len(archive_x) > 0:
            mu1 = np.sqrt(np.sum(np.power(x - archive_x, 2), 1)) / (max_eig_sqrt * sigma)  # L / (mu_1 * sigma_mean)
            criticality = norm.cdf(mu1 + archive_d) - norm.cdf(mu1 - archive_d)
            return np.argsort(-1 * criticality)[np.sort(-1 * criticality) < -1 * self.c_threshold]
        return np.empty((0,))

    def is_new_basin(self, x1, x4, f1, f4, args=None):
        new_basin, n = False, 0
        if np.sqrt(np.sum(np.power(x1 - x4, 2))) > 0:
            max_f = np.maximum(f1, f4)
            direction = x4 - x1  # search direction
            delta = np.sqrt(np.sum(np.power(direction, 2)))
            direction = direction / delta / 2.618
            x2 = x1 + delta*direction
            f2 = self._evaluate_fitness(x2, args)
            n += 1
            if f2 > max_f:
                new_basin = True
            elif n < self.budget:  # Max evaluation budget for the hill-valley ( Detect Multimodal) function
                x3 = x1 + 1.618*delta*direction
                f3 = self._evaluate_fitness(x3, args)
                n += 1
                if f3 > max_f:
                    new_basin = True
                else:
                    while (n < self.budget) and (np.sqrt(np.sum(np.power(x1 - x4, 2))) > 0):
                        if f2 < f3:
                            delta = np.sqrt(np.sum(np.power(x4 - x2, 2))) / 2.618
                            x1, f1 = np.copy(x2), np.copy(f2)
                            x2, f2 = np.copy(x3), np.copy(f3)
                            x3 = x1 + 1.618*delta*direction
                            f3 = self._evaluate_fitness(x3, args)
                            n += 1
                        elif f2 > f3:
                            delta = np.sqrt(np.sum(np.power(x3 - x1, 2))) / 2.618
                            x4, f4 = np.copy(x3), np.copy(f3)
                            x3, f3 = np.copy(x2), np.copy(f2)
                            x2 = x1 + delta*direction
                            f2 = self._evaluate_fitness(x2, args)
                            n += 1
                        else:
                            delta = np.sqrt(np.sum(np.power(x3 - x2, 2))) / 2.618034
                            x1, f1 = np.copy(x2), np.copy(f2)
                            x4, f4 = np.copy(x3), np.copy(f3)
                            x2 = x1 + delta*direction
                            x3 = x4 - delta*direction
                            f2 = self._evaluate_fitness(x2, args)
                            f3 = self._evaluate_fitness(x3, args)
                            n += 2
                        if np.maximum(f2, f3) > max_f:
                            new_basin = True
                            break
        return new_basin

    def update_population_size(self, i_used):
        factor = (self.max_function_evaluations - self.n_function_evaluations) / (
                self.n_individuals * self._n_s_bak * i_used)
        if factor >= 2:
            self.n_individuals = 2 * self.n_individuals
            self.n_s = np.copy(self._n_s_bak)
        elif factor >= 1:
            self.n_individuals = int(self.n_individuals ** 2 * factor)
            self.n_s = np.copy(self._n_s_bak)
        else:
            self.n_s = np.max([1, int(self._n_s_bak * factor)])
        self.n_parents = int(self.n_individuals / 2)
        w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
        self._w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
        self._mu_eff = 1 / np.sum(np.power(self._w, 2))
        self.c_cov = self.n_parents / (self.n_parents + self.ndim_problem * (self.ndim_problem + 1))  # 1/tau_c
        self.n_elt = np.maximum(1, int(0.15 * self.n_individuals))  # number of elites

    def update_archive(self, yy, yy_f, archive_x, archive_f, archive_d, args=None):
        hat_d = self.hat_d_0
        if len(archive_f) > 0:
            if (np.min(yy_f) + self.fitness_threshold) < np.min(archive_f):  # discard archive solutions
                archive_x, archive_f, archive_d = np.empty((0, self.ndim_problem)), np.empty((0,)), np.empty((0,))
            else:
                hat_d = np.percentile(archive_d, self.p)

        # Consider only global minima, among the recently generated solutions
        index = np.arange(len(yy_f))[yy_f <= self.fitness_threshold + np.min(np.hstack((archive_f, yy_f)))]
        n_rep = np.zeros((len(archive_x),))  # number of subpopulations that have converged to corresponding basin
        for i in index:
            y, f_y = yy[i], yy_f[i]
            archive_x = archive_x[
                np.argsort(np.array([np.sqrt(np.sum(np.power(y - x, 2))) for x in archive_x]))]  # sort
            is_new = 1
            for j in range(len(archive_x)):
                x, f_x = archive_x[j], archive_f[j]
                is_basin = True
                if (self.max_function_evaluations - self.n_function_evaluations) > self.budget:
                    is_basin = self.is_new_basin(y, x, f_y, f_x, args)
                if is_basin:  # share the same basin
                    if f_y < f_x:
                        archive_x[j], archive_f[j] = y, f_y
                    n_rep[j] += 1
                    is_new = 0  # not a new basin
                    break
            if is_new == 1:
                archive_x = np.vstack((archive_x, y))
                archive_f = np.hstack((archive_f, f_y))
                archive_d = np.hstack((archive_d, hat_d))
                n_rep = np.hstack((n_rep, 0))

        diff = n_rep - self.alpha_new * (len(index) / len(archive_x))
        for i in range(len(archive_x)):
            if diff[i] > 0:
                archive_d[i] *= (1 + diff[i])**self.tau_hat_d
            else:
                archive_d[i] *= (1 - diff[i])**(-1 * self.tau_hat_d)
        return archive_x, archive_f, archive_d

    def generate_subpopulation(self, mean, cov, sigma, superior_mean, superior_d,
                               archive_x, archive_f, archive_d, elt_x, elt_f, elt_z, elt_s,
                               best_f_ne, median_f_ne, args=None):
        """
        elt_x: initial elite solutions of the subpopulation
        elt_f: initial elite function fitness
        elt_z: Initial value of the elite variation vector
        elt_s: Initial value of the elite global step sizes
        best_f_ne: History of the best of non-elite solutions for each subpopulation
        med_f_ne: History of the median of non-elite solutions for each subpopulation
        """
        s = 1  # for temporary shrinkage of the taboo regions
        w, v = np.linalg.eig(cov)
        inv_cov = np.matmul(np.matmul(v, np.diag(1 / w)), v.T)  # Inverse of the matrix
        sqrt_w = np.sqrt(w)

        # Generate taboo points
        taboo_points = np.copy(superior_mean)  # Center of fitter subpopulations
        taboo_points_d = np.copy(superior_d)  # The normalized taboo distance of fitter subpopulations
        for i in range(len(archive_x)):  # Consider fitter archived points as taboo points
            base = np.ones((len(elt_f),))[archive_f[i] < elt_f]
            if len(base) == len(elt_f):
                taboo_points = np.vstack((taboo_points, archive_x[i]))
                taboo_points_d = np.hstack((taboo_points_d, archive_d[i]))

        # Determine which taboo points are critical (numpy-ndarray)
        c_index = self.find_critical_points_descend_index(mean, sigma, np.max(sqrt_w), taboo_points, taboo_points_d)

        # Sample, and Generate \lambda taboo acceptable solutions
        x = np.empty((self.n_individuals, self.ndim_problem))
        f = np.empty((self.n_individuals,))
        sigmas = np.zeros((self.n_individuals,))
        z = np.copy(x)
        for n in range(self.n_individuals):
            accept = False
            while not accept:
                sigmas[n] = sigma * np.exp(self.rng_optimization.standard_normal() * self.tau)  # for Fig. 2. (R1)
                sqrt_cov = np.matmul(v, np.diag(sqrt_w))
                z[n] = np.dot(sqrt_cov, self.rng_optimization.standard_normal((self.ndim_problem,)))  # for Fig. 2. (R2)
                z[n] = sigmas[n] * z[n]  # for Fig. 2. (R3)
                x[n] = mean + z[n]  # for Fig. 2. (R4)
                if len(c_index) == 0:
                    accept = True
                for c in c_index:
                    d = calculate_mahalanobis_distance(x[n], taboo_points[c], inv_cov)
                    accept = d > sigma * taboo_points_d[c] * s
                    if not accept:  # reject
                        s *= self.c_red  # Temporary shrink the size of the taboo regions
                        break
            f[n] = self._evaluate_fitness(x[n], args)

        # best for the non-elite solutions
        best_f_ne = np.hstack((best_f_ne, np.min(f)))
        median_f_ne = np.hstack((median_f_ne, np.median(f)))

        # Append the surviving elites from the previous generation (Recombine)
        for k in range(len(elt_f)):
            if elt_f[k] < self._best_so_far_y_bak:
                accept = False
                for c in c_index:
                    d = calculate_mahalanobis_distance(elt_x[k], taboo_points[c], inv_cov)
                    accept = d > sigma * taboo_points_d[c] * s
                    if not accept:
                        break
                if (len(c_index) == 0) or accept:  # The elite was not in a taboo region
                    x = np.vstack((x, elt_x[k]))
                    f = np.hstack((f, elt_f[k]))
                    sigmas = np.hstack((sigmas, elt_s[k]))
                    z = np.vstack((z, elt_z[k]))

        # Update parameters of the subpopulation using Equation 6
        order = np.argsort(f)
        mean = np.sum(self._w.reshape(self.n_parents, 1) * x[order[:self.n_parents]], 0)
        c = 0
        for i in range(self.n_parents):
            m1, m2 = np.meshgrid(z[order[i]], z[order[i]])
            c += self._w[i] * (m2 * m1)
        cov = (1 - self.c_cov) * cov + self.c_cov * c
        cov = (cov + cov.T) / 2  # A symmetric matrix
        sigma *= np.exp(np.dot(self._w, np.log(sigmas[:self.n_parents]))) / np.exp(np.mean(np.log(sigmas)))

        # Update the elite solutions
        elt_x = x[order[:self.n_elt]]
        elt_f = f[order[:self.n_elt]]
        elt_z = z[order[:self.n_elt]]
        elt_s = sigmas[order[:self.n_elt]]
        elt_x_f_z_s = (elt_x, elt_f, elt_z, elt_s)
        mean_cov_sigma = (mean, cov, sigma)
        best_x_f = (x[order[0]], f[order[0]])
        best_median_f_ne = (best_f_ne, median_f_ne)
        return best_x_f, mean_cov_sigma, elt_x_f_z_s, best_median_f_ne

    def initialize_subpopulations(self, archive_xx, archive_d):
        hat_d = self.hat_d_0
        if len(archive_d) > 0:
            hat_d = np.percentile(archive_d, self.p)

        base = (self.initial_upper_boundary - self.initial_lower_boundary)**2
        cov, inv_cov, sigma = np.diag(base), np.diag(1 / base), self.sigma
        means = np.empty((self.n_s, self.ndim_problem))
        i, n_rej = 0, 0
        while i < self.n_s:
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
            accept = True
            for j in range(len(archive_xx)):  # for archived points
                if (2*hat_d + archive_d[j])*sigma > calculate_mahalanobis_distance(archive_xx[j], x, inv_cov):
                    accept = False
                    break
            if accept:
                for k in range(i):  # for previously sampled points
                    if 3*hat_d*sigma > calculate_mahalanobis_distance(means[k], x, inv_cov):
                        accept = False
                        break
            if accept:
                means[i], n_rej = x, 0
                i += 1
            else:
                n_rej += 1
            if n_rej > 100:
                sigma, n_rej = self.c_red * sigma, 0  # shrink the size of each taboo region
        means_d = hat_d * np.ones((self.n_s,))
        sigmas = sigma * np.ones((self.n_s,))
        return means, means_d, sigmas, cov

    def iterate(self, means=None, means_d=None, sigmas=None, cov=None,
                archive_x=None, archive_f=None, archive_d=None, args=None):
        n1 = 120 + int(30 * self.ndim_problem / self.n_individuals)  # No improvement
        n2 = 10 + int(30 * self.ndim_problem / self.n_individuals)  # Stalled size (stagnation)
        elt_x = {k: np.tile(means[k], (self.n_elt, 1)) for k in range(self.n_s)}
        elt_z = {k: np.zeros((self.n_elt, self.ndim_problem)) for k in range(self.n_s)}
        elt_f = {k: np.ones((self.n_elt,)) * self._best_so_far_y_bak for k in range(self.n_s)}
        elt_s = {k: np.ones((self.n_elt,)) * np.mean(sigmas) for k in range(self.n_s)}
        cov = {k: cov for k in range(self.n_s)}
        best_f_ne = {k: np.empty((0,)) for k in range(self.n_s)}  # no elite
        median_f_ne = {k: np.empty((0,)) for k in range(self.n_s)}
        superior_index = {k: np.arange(k) for k in range(self.n_s)}
        best_x = np.copy(means)
        best_f = np.ones((self.n_s,)) * self._best_so_far_y_bak

        ap = np.arange(self.n_s)  # Activating subpopulations
        tp = []  # Terminated subpopulations
        n_u = np.ones((self.n_s,))  # used iteration number
        terminate_flag, n_iteration = False, 0

        while (len(ap) > 0) and (not terminate_flag):
            for m in ap:
                # The number of potential global minima in the current restart. It is used only for termination of the
                # restart so that sufficient budget remains for analyzing the subpopulations later.
                # This is a specialization for GECCO2016 competition
                cnt = np.ones((self.n_s,))[(best_f - self.fitness_threshold) <= min(np.hstack((archive_f, best_f)))]
                cnt = 1 + np.sum(cnt)
                if (self.max_function_evaluations - self.n_function_evaluations) <= \
                        (cnt * self.budget * (len(archive_x) + cnt / 2) + self.n_individuals):
                    self.n_function_evaluations = self.max_function_evaluations
                    terminate_flag = True  # terminate the restart
                    break

                flag = [np.linalg.cond(cov[m]) >= self.n_condition, self._check_terminations(), False, False, False]
                # Termination condition
                if n_iteration >= n2:
                    flag[2] = np.max(best_f_ne[m][-n2:]) - np.min(best_f_ne[m][-n2:]) < self.fitness_threshold
                if n_iteration >= n1:
                    min_best = np.median(best_f_ne[m][-20:]) - np.median(best_f_ne[m][-n1:(-n1 + 20)])
                    min_median = np.median(median_f_ne[m][-20:]) - np.median(median_f_ne[m][-n1:(-n1 + 20)])
                    flag[3] = min_best >= 0
                    flag[4] = min_median >= 0
                if any(flag):  # Terminate the subpopulation if a stopping criterion is satisfied
                    means[m] = best_x[m]
                    tp.append(m)
                    continue

                n_u[m] = n_iteration + 1
                superior_mean = means[superior_index[m]]
                superior_d = means_d[superior_index[m]]
                best_xx_f, mean_cov_sigma, elt_x_f_s_z, best_median_f_ne = \
                    self.generate_subpopulation(means[m], cov[m], sigmas[m], superior_mean, superior_d, archive_x,
                                                archive_f, archive_d, elt_x[m], elt_f[m], elt_z[m], elt_s[m],
                                                best_f_ne[m], median_f_ne[m], args)
                best_x[m], best_f[m] = best_xx_f
                means[m], cov[m], sigmas[m] = mean_cov_sigma
                elt_x[m], elt_f[m], elt_z[m], elt_s[m] = elt_x_f_s_z
                best_f_ne[m], median_f_ne[m] = best_median_f_ne

            # Find non-terminated subpopulations
            ap = np.array([i for i in ap if i not in tp])
            if len(ap) > 0:
                ap = ap[np.argsort(best_f[ap])]
                base = np.arange(len(best_f))
                superior_index.update({m: base[best_f < best_f[m]] for m in ap})
            n_iteration += 1
            n1 = 120 + int(0.2 * n_iteration + 30 * self.ndim_problem / self.n_individuals)
        return best_x, best_f, np.mean(n_u)

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        archive_x, archive_f, archive_d = self.initialize(args)
        while True:
            # Initialize subpopulations (Algorithm 2)
            means, means_d, sigmas, cov = self.initialize_subpopulations(archive_x, archive_d)

            # Iterate
            best_x, best_f, u = self.iterate(means, means_d, sigmas, cov, archive_x, archive_f, archive_d, args)

            # Update archive (Algorithm 1)
            archive_x, archive_f, archive_d = self.update_archive(best_x, best_f, archive_x, archive_f, archive_d)
            if self.record_fitness:
                fitness.extend(best_f)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(best_f)
            # Restart (Equations 8)
            self.update_population_size(u)
        return self._collect_results(fitness)
