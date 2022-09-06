import numpy as np
from scipy.stats import norm

from pypop7.optimizers.es.es import ES


def mahalanobis_distance(y, x, inv_cov):
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
        self.tau = np.sqrt(0.5 / self.ndim_problem)  # the learning rate of global step size
        self.c_cov = self.n_parents / (self.n_parents + self.ndim_problem * (self.ndim_problem + 1))  # 1/tau_c
        self.n_elt = np.maximum(1, int(0.15 * self.n_individuals))  # number of elites
        self.tau_hat_d = np.sqrt(1 / self.ndim_problem)  # the learning rate of the normalized taboo distance
        self.c_red = np.power(0.99, 1 / self.ndim_problem)  # the size of each taboo region shrinks by (1-self.c_red)
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

    def index_critical_points(self, x, sigma, max_eig_sqrt, taboo_points, taboo_points_d):
        if len(taboo_points) > 0:
            mu1 = np.sqrt(np.sum(np.power(x - taboo_points, 2), 1)) / (max_eig_sqrt * sigma)  # L / (mu_1 * sigma_mean)
            criticality = norm.cdf(mu1 + taboo_points_d) - norm.cdf(mu1 - taboo_points_d)
            return np.argsort(-1 * criticality)[np.sort(-1 * criticality) < -1 * self.c_threshold]
        return np.empty((0,))

    def is_new_basin(self, x1, f1, x4, f4, args=None):
        new_basin, n = False, 0
        if np.sqrt(np.sum(np.power(x1 - x4, 2))) > 0:
            max_f = np.maximum(f1, f4)
            direction = x4 - x1  # search direction
            delta = np.sqrt(np.sum(np.power(direction, 2)))
            direction = direction / delta / 2.618
            x2 = x1 + delta * direction
            f2 = self._evaluate_fitness(x2, args)
            n += 1
            if f2 > max_f:
                new_basin = True
            elif n < self.budget:  # Max evaluation budget for the hill-valley ( Detect Multimodal) function
                x3 = x1 + 1.618 * delta * direction
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
                            x3 = x1 + 1.618 * delta * direction
                            f3 = self._evaluate_fitness(x3, args)
                            n += 1
                        elif f2 > f3:
                            delta = np.sqrt(np.sum(np.power(x3 - x1, 2))) / 2.618
                            x4, f4 = np.copy(x3), np.copy(f3)
                            x3, f3 = np.copy(x2), np.copy(f2)
                            x2 = x1 + delta * direction
                            f2 = self._evaluate_fitness(x2, args)
                            n += 1
                        else:
                            delta = np.sqrt(np.sum(np.power(x3 - x2, 2))) / 2.618034
                            x1, f1 = np.copy(x2), np.copy(f2)
                            x4, f4 = np.copy(x3), np.copy(f3)
                            x2 = x1 + delta * direction
                            x3 = x4 - delta * direction
                            f2 = self._evaluate_fitness(x2, args)
                            f3 = self._evaluate_fitness(x3, args)
                            n += 2
                        if np.maximum(f2, f3) > max_f:
                            new_basin = True
                            break
        return new_basin

    def update_population_size(self, average_used_iteration_number):
        factor = (self.max_function_evaluations - self.n_function_evaluations) / (
                self.n_individuals * self._n_s_bak * average_used_iteration_number)
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

    def update_archive(self, best_x, best_y, archive_x, archive_y, archive_d, args=None):
        hat_d = self.hat_d_0 if len(archive_d) == 0 else np.percentile(archive_d, self.p)
        if len(archive_y) > 0:
            if (np.min(best_y) + self.fitness_threshold) < np.min(archive_y):  # discard archive solutions
                archive_x, archive_y, archive_d = np.empty((0, self.ndim_problem)), np.empty((0,)), np.empty((0,))

        # Consider only global minima, among the recently generated solutions
        index = np.arange(self.n_s)[best_y <= (self.fitness_threshold + np.min(np.hstack((archive_y, best_y))))]
        n_rep = np.zeros((len(archive_x),))  # number of subpopulations that have converged to corresponding basin
        for i in index:
            archive_x = archive_x[np.argsort([np.sqrt(np.sum(np.power(best_x[i] - x, 2))) for x in archive_x])]  # sort
            is_new = 1
            for j in range(len(archive_x)):
                is_basin = self.is_new_basin(best_x[i], best_y[i], archive_x[j], archive_y[j], args)
                if self._check_terminations():
                    return archive_x, archive_y, archive_d

                if is_basin:  # share the same basin
                    if best_y[i] < archive_y[j]:
                        archive_x[j], archive_y[j] = best_x[i], best_y[i]
                    n_rep[j] += 1
                    is_new = 0  # not a new basin
                    break
            if is_new == 1:
                archive_x = np.vstack((archive_x, best_x[i]))
                archive_y = np.hstack((archive_y, best_y[i]))
                archive_d = np.hstack((archive_d, hat_d))
                n_rep = np.hstack((n_rep, 0))

        diff = n_rep - self.alpha_new * (len(index) / len(archive_x))
        for i in range(len(archive_x)):
            if diff[i] > 0:
                archive_d[i] *= (1 + diff[i]) ** self.tau_hat_d
            else:
                archive_d[i] *= (1 - diff[i]) ** (-1 * self.tau_hat_d)
        return archive_x, archive_y, archive_d

    def initialize_subpopulation(self, archive_xx, archive_d):
        hat_d = self.hat_d_0 if len(archive_d) == 0 else np.percentile(archive_d, self.p)
        base = (self.initial_upper_boundary - self.initial_lower_boundary) ** 2
        cov, inv_cov, sigma = np.diag(base), np.diag(1 / base), self.sigma
        means = np.empty((self.n_s, self.ndim_problem))
        for i in range(self.n_s):
            n_rej = 0
            while True:
                accept = True
                x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
                for j in range(len(archive_xx)):  # for archived points
                    accept = mahalanobis_distance(archive_xx[j], x, inv_cov) >= ((2 * hat_d + archive_d[j]) * sigma)
                    if not accept:
                        break
                if accept:
                    for k in range(i):  # for previously sampled points
                        accept = mahalanobis_distance(means[k], x, inv_cov) >= (3 * hat_d * sigma)
                        if not accept:
                            break
                if accept:
                    means[i] = x
                    break
                else:
                    n_rej += 1
                if n_rej > 100:
                    sigma, n_rej = self.c_red * sigma, 0  # shrink the size of each taboo region
        sigmas = sigma * np.ones((self.n_s,))
        means_d = hat_d * np.ones((self.n_s,))
        return means, sigmas, cov, means_d

    def iterate_subpopulation(self, mean, cov, sigma, elt_x, elt_y, elt_z, elt_s,
                              best_y_ne, median_y_ne, superior_mean, superior_d,
                              archive_x, archive_y, archive_d, args):
        # Singular value decomposition
        u, s, v = np.linalg.svd(cov)
        inv_cov = v @ np.diag(1 / s) @ u  # Inverse of the matrix
        sqrt_w = np.sqrt(s)
        sqrt_cov = u @ np.diag(sqrt_w)

        # Generate taboo points
        taboo, taboo_d = self.generate_taboo_point(superior_mean, superior_d, archive_x, archive_y, archive_d, elt_y)

        # Determine critical taboo points
        c_index = self.index_critical_points(mean, sigma, np.max(sqrt_w), taboo, taboo_d)

        # Sample
        x, y, z, sigmas, scale = self.sample(mean, sigma, sqrt_cov, inv_cov, taboo, taboo_d, c_index, args)

        # for the non-elite solutions
        best_y_ne = np.hstack((best_y_ne, np.min(y)))
        median_y_ne = np.hstack((median_y_ne, np.median(y)))

        x, y, z, sigmas = self.recombine(
            inv_cov, sigma, x, y, z, sigmas, scale, taboo, taboo_d, c_index, elt_x, elt_y, elt_z, elt_s)

        mean, cov, sigma = self._update_distribution(x, z, y, cov, sigma, sigmas)

        order = np.argsort(y)
        best_x, best_y = x[order[0]], y[order[0]]
        base = order[:self.n_elt]  # Update the elite solutions
        elt_x, elt_y, elt_z, elt_s = x[base], y[base], z[base], sigmas[base]
        return best_x, best_y, mean, cov, sigma, elt_x, elt_y, elt_z, elt_s, best_y_ne, median_y_ne

    def recombine(self, inv_cov, sigma, x, y, z, sigmas, scale, taboo, taboo_d, c_index, elt_x, elt_y, elt_z, elt_s):
        # Append the surviving elites from the previous generation
        for k in range(len(elt_y)):
            if elt_y[k] < self._best_so_far_y_bak:
                accept = False
                for c in c_index:
                    d = mahalanobis_distance(elt_x[k], taboo[c], inv_cov)
                    accept = d > sigma * taboo_d[c] * scale
                    if not accept:
                        break
                if accept:
                    x = np.vstack((x, elt_x[k]))
                    y = np.hstack((y, elt_y[k]))
                    z = np.vstack((z, elt_z[k]))
                    sigmas = np.hstack((sigmas, elt_s[k]))
        return x, y, z, sigmas

    def generate_taboo_point(self, superior_mean, superior_d, archive_x, archive_y, archive_d, elt_y):
        # Generate taboo points
        taboo = np.copy(superior_mean)  # Center of fitter subpopulations
        taboo_d = np.copy(superior_d)  # The normalized taboo distance of fitter subpopulations
        for i in range(len(archive_x)):  # Consider fitter archived points as taboo points
            base = np.ones((self.n_elt,))[archive_y[i] < elt_y]
            if len(base) == self.n_elt:
                taboo = np.vstack((taboo, archive_x[i]))
                taboo_d = np.hstack((taboo_d, archive_d[i]))
        return taboo, taboo_d

    def sample(self, mean, sigma, sqrt_cov, inv_cov, taboo, taboo_d, c_index, args=None):
        x = np.empty((self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        z = np.empty((self.n_individuals, self.ndim_problem))
        sigmas = np.empty((self.n_individuals,))
        scale = 1  # for temporary shrinkage of the taboo regions
        for n in range(self.n_individuals):
            accept = False
            while not accept:
                sigmas[n] = sigma * np.exp(self.rng_optimization.standard_normal() * self.tau)  # for Fig. 2. (R1)
                z[n] = np.dot(sqrt_cov, self.rng_optimization.standard_normal((self.ndim_problem,)))  # for Fig. 2. (R2)
                z[n] = sigmas[n] * z[n]  # for Fig. 2. (R3)
                x[n] = mean + z[n]  # for Fig. 2. (R4)
                if len(c_index) == 0:
                    break
                for c in c_index:
                    d = mahalanobis_distance(x[n], taboo[c], inv_cov)
                    accept = d > sigma * taboo_d[c] * scale
                    if not accept:  # reject
                        scale *= self.c_red  # Temporary shrink the size of the taboo regions
                        break
            y[n] = self._evaluate_fitness(x[n], args)
        return x, y, z, sigmas, scale

    def _update_distribution(self, x=None, z=None, y=None, cov=None, sigma=None, sigmas=None):
        # Update parameters of the subpopulation using Equation 6
        order = np.argsort(y)
        mean = np.sum(self._w.reshape(self.n_parents, 1) * x[order[:self.n_parents]], 0)
        c = np.zeros((self.ndim_problem, self.ndim_problem))
        for i in range(self.n_parents):
            m1, m2 = np.meshgrid(z[order[i]], z[order[i]])
            c += self._w[i] * (m2 * m1)
        cov = (1 - self.c_cov) * cov + self.c_cov * c
        cov = (cov + cov.T) / 2  # A symmetric matrix
        sigma *= np.exp(np.dot(self._w, np.log(sigmas[:self.n_parents]))) / np.exp(np.mean(np.log(sigmas)))
        return mean, cov, sigma

    def iterate(self, means=None, sigmas=None, cov=None, means_d=None,
                archive_x=None, archive_y=None, archive_d=None, args=None):
        n1 = 120 + int(30 * self.ndim_problem / self.n_individuals)  # no improvement
        n2 = 10 + int(30 * self.ndim_problem / self.n_individuals)  # stagnation
        cov_s = {k: cov for k in range(self.n_s)}
        elt_x = {k: np.tile(means[k], (self.n_elt, 1)) for k in range(self.n_s)}
        elt_y = {k: np.ones((self.n_elt,)) * self._best_so_far_y_bak for k in range(self.n_s)}
        elt_z = {k: np.zeros((self.n_elt, self.ndim_problem)) for k in range(self.n_s)}
        elt_s = {k: np.ones((self.n_elt,)) * np.mean(sigmas) for k in range(self.n_s)}
        best_y_ne = {k: np.empty((0,)) for k in range(self.n_s)}
        median_y_ne = {k: np.empty((0,)) for k in range(self.n_s)}
        superior_index = {k: np.arange(k) for k in range(self.n_s)}
        best_x = np.copy(means)
        best_y = np.ones((self.n_s,)) * self._best_so_far_y_bak
        ap = np.arange(self.n_s)  # activating subpopulations
        tp = []  # terminated subpopulations
        n_u = np.ones((self.n_s,))  # used iteration number

        n_iteration = 0
        while len(ap) > 0:
            for m in ap:
                # Termination subpopulation condition
                flag = [np.linalg.cond(cov_s[m]) >= self.n_condition, False, False, False]
                if n_iteration >= n2:
                    flag[1] = (np.max(best_y_ne[m][-n2:]) - np.min(best_y_ne[m][-n2:])) < self.fitness_threshold
                if n_iteration >= n1:
                    flag[2] = (np.median(best_y_ne[m][-20:]) - np.median(best_y_ne[m][-n1:(-n1 + 20)])) >= 0
                    flag[3] = (np.median(median_y_ne[m][-20:]) - np.median(median_y_ne[m][-n1:(-n1 + 20)])) >= 0
                if any(flag):
                    means[m] = best_x[m]
                    tp.append(m)
                    n_u[m] = n_iteration + 1
                    continue

                best_x[m], best_y[m], \
                    means[m], cov_s[m], sigmas[m], elt_x[m], elt_y[m], elt_z[m], elt_s[m], \
                    best_y_ne[m], median_y_ne[m] = \
                    self.iterate_subpopulation(
                        means[m], cov_s[m], sigmas[m], elt_x[m], elt_y[m], elt_z[m], elt_s[m],
                        best_y_ne[m], median_y_ne[m],
                        means[superior_index[m]], means_d[superior_index[m]],
                        archive_x, archive_y, archive_d, args)

                if self._check_terminations():
                    return best_x, best_y, np.mean(n_u)

            # Find non-terminated subpopulations
            ap = np.setdiff1d(ap, tp)
            if len(ap) > 0:
                ap = ap[np.argsort(best_y[ap])]
                base = np.arange(self.n_s)
                superior_index.update({m: base[best_y < best_y[m]] for m in ap})
            n_iteration += 1
            n1 = 120 + int(0.2 * n_iteration + 30 * self.ndim_problem / self.n_individuals)
        return best_x, best_y, np.mean(n_u)

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        archive_x, archive_y, archive_d = self.initialize(args)
        while True:
            means, sigmas, cov, means_d = self.initialize_subpopulation(archive_x, archive_d)  # (Algorithm 2)
            best_x, best_y, n_u = self.iterate(means, sigmas, cov, means_d, archive_x, archive_y, archive_d, args)
            # Update archive (Algorithm 1)
            archive_x, archive_y, archive_d = self.update_archive(best_x, best_y, archive_x, archive_y, archive_d)
            if self.record_fitness:
                fitness.extend(best_y)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(best_y)
            self.update_population_size(n_u)  # Restart (Equations 8)
        return self._collect_results(fitness)
