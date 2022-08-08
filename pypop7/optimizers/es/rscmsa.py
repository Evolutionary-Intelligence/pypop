from abc import ABC

import numpy as np
import math
from scipy.stats import norm

from pypop7.optimizers.es.es import ES


# (2.3.1 Distance Metrix)
def calculate_mahalanobis_distance(yy, x, inv_cov):
    if (yy.ndim == 0) and (inv_cov.ndim == 0):  # yy is a scalar
        return np.sqrt(np.dot(np.dot(yy - x, inv_cov), (yy - x).T))
    elif (yy.ndim == 1) and (inv_cov.ndim == 2):  # yy is a vector
        return np.sqrt(np.dot(np.dot(yy - x, inv_cov), (yy - x).T))
    elif (yy.ndim == 2) and (inv_cov.ndim == 2):  # yy is a two-dimensional matrix
        return np.array([np.sqrt(np.dot(np.dot(y - x, inv_cov), (y - x).T)) for y in yy])
    else:
        raise TypeError(f'{yy} and {inv_cov}.')


class RSCMSA(ES, ABC):
    """Covariance Matrix Self-Adaptation Evolution Strategy with Repelling Subpopulations (RS-CMSA)

    Reference
    ---------
    Ahrari, A., Deb, K., and Preuss, M. 2017.
    Multimodal optimization by covariance matrix self-adaptation evolution strategy with repelling subpopulations.
    Evolutionary Computation, 25(3), 439–471.
    https://doi.org/10.1162/evco_a_00182
    """

    def __init__(self, problem, options):
        ES.__init__(self, problem, options)

        self.tau = np.sqrt(0.5 / self.ndim_problem)  #
        # the default value of the normalized taboo distance, which is assigned to the new members of Archive as well
        # as the subpopulations during the restart.
        self._hat_d_bak = 1  # default value of the normalized taboo distance
        self.tau_hat_d = np.sqrt(1 / self.ndim_problem)  # the learning rate of the normalized taboo distance
        self.alpha_new = 0.5  # target rate of basin identification
        # the size of each taboo region shrinks by (1-self.c_red)
        # A smaller value of self.c_red speeds up reduction of the normalized taboo distances
        self.c_red = np.power(0.99, 1 / self.ndim_problem)
        self.criticality_threshold = 0.01  # criticality threshold for taboo points
        self.n_s = options.get('n_subpopulations', 10)  # number of subpopulations
        self._n_s_bak = np.copy(self.n_s)  # default number of subpopulations
        self.percentile = 25  # Percentile for the normalized taboo distance of the current subpopulations
        self.max_budget = 10  # Max evaluation budget for the hill-valley ( Detect Multimodal) function
        self.distance_metric = options.get('distance_metric', 'Mahalanobis')
        if self.distance_metric not in ['Euclidean', 'Mahalanobis']:
            raise TypeError(f'Distance metric must be "Euclidean" or "Mahalanobis" not {self.distance_metric}.')
        self.best_so_far_x_bak = np.copy(self.best_so_far_x)
        self.best_so_far_y_bak = np.copy(self.best_so_far_y)
        self.n_condition = 1e14

        # the adaptation interval constant for Covariance Matrix
        # (1/self.tau_c) can be interpreted as the learning rate for the Covariance Matrix
        self.tau_c = 1 + self.ndim_problem * (self.ndim_problem + 1) / self.n_parents
        # The n_elt best solutions of this iteration will survive to the next iteration
        self.n_elt = max(1, math.floor(0.15 * self.n_individuals))  # number of elites (2.3.3)
        # average number of iterations used by the subpopulations in the previous restart
        self.average_iteration_subpopulations = 1

    # (2.6.3 Calculation of the Mean Rejection Probability)
    def find_critical_points_descend_index(self, x, sigma, inv_cov, archive_xx, archive_d):
        # Return the index of critical taboo points sorted based on criticality (descend)
        if (archive_xx.ndim == 2) and (archive_xx.size > 0):
            mu = np.sqrt(min(np.abs(np.linalg.eigvals(inv_cov)))) / sigma
            mu = mu * np.sum(np.power(x - archive_xx, 2), 1) - archive_d
            criticality = 1 - 2 * norm.cdf(mu)  # standard normal distribution
            return np.argsort(-1 * criticality)[np.sort(-1 * criticality) < -1 * self.criticality_threshold]
        return np.empty((0,))

    def is_new_basin(self, x1, x4, f1, f4, args=None):
        new_basin, n = False, 0
        if np.sqrt(np.sum(np.power(x1 - x4, 2))) > 0:
            max_f = max(f1, f4)
            direction = x4 - x1  # search direction
            delta = np.sqrt(np.sum(np.power(x4 - x1, 2)))
            direction = direction / delta / 2.618
            x2 = x1 + delta * direction
            f2 = self._evaluate_fitness(x2, args)
            n += 1
            if f2 > max_f:
                new_basin = True
            elif n < self.max_budget:  # Max evaluation budget for the hill-valley ( Detect Multimodal) function
                x3 = x1 + 1.618 * delta * direction
                f3 = self._evaluate_fitness(x3, args)
                n += 1
                if f3 > max_f:
                    new_basin = True
                else:
                    while (n < self.max_budget) and (np.sqrt(np.sum(np.power(x1 - x4, 2))) > 0):
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
                        if max(f2, f3) > max_f:
                            new_basin = True
                            break
        return new_basin

    # (2.4 Restarts with Increasing Population Size)
    def update_population_size(self):
        factor = (self.max_function_evaluations - self.n_function_evaluations) / (
                self.n_individuals * self._n_s_bak * self.average_iteration_subpopulations)
        if factor >= 2:
            self.n_s = np.copy(self._n_s_bak)
            self.n_individuals = 2 * self.n_individuals
        elif factor >= 1:
            self.n_s = np.copy(self._n_s_bak)
            self.n_individuals = math.floor(self.n_individuals * factor)
        else:
            self.n_s = max(1, math.floor(self._n_s_bak * factor))

        self.sigma = np.copy(self._sigma_bak)
        self.n_parents = int(self.n_individuals / 2)
        w_base, w = np.log((self.n_individuals + 1) / 2), np.log(np.arange(self.n_parents) + 1)
        self._w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
        self._mu_eff = 1 / np.sum(np.power(self._w, 2))

        # the adaptation interval constant for Covariance Matrix
        # (1/self.tau_c) can be interpreted as the learning rate for the Covariance Matrix
        self.tau_c = 1 + self.ndim_problem * (self.ndim_problem + 1) / self.n_parents
        # The n_elt best solutions of this iteration will survive to the next iteration
        self.n_elt = max(1, math.floor(0.15 * self.n_individuals))  # number of elites (2.3.3)
        # average number of iterations used by the subpopulations in the previous restart
        self.average_iteration_subpopulations = 1

    # Algorithm 1 (Updating Archive and the normalized taboo distances)
    # (2.5 Adaptation of the Normalized Taboo Distance)
    # Update the archive using the best solutions of the converged subpopulations (yy) and the old archive
    def updating_archive(self, yy, yy_f, archive_xx, archive_f, archive_d, args=None):
        hat_d_0 = self._hat_d_bak  # If archive is empty, then hat_d_0 = 1
        if (archive_f.size > 0) and (archive_d.size > 0):
            if min(yy_f) + self.fitness_threshold < min(archive_f):  # not global minimum, discard them
                archive_xx, archive_f, archive_d = np.empty((0, self.ndim_problem)), np.empty((0,)), np.empty((0,))
            else:
                hat_d_0 = np.percentile(archive_d, self.percentile)

        # Consider only global minima, among the recently generated solutions
        idx = np.arange(yy_f.size)[yy_f <= self.fitness_threshold + np.min(np.hstack((archive_f, yy_f)))]

        # the number of subpopulations that have converged to the corresponding basin
        n_rep = np.empty((len(archive_xx),))
        for i in idx:
            y, f_y = yy[i], yy_f[i]
            # Sort the points in Archive based on their Euclidean distance to y
            archive_xx = archive_xx[np.argsort(np.array([np.linalg.norm(y - x, ord=2) for x in archive_xx]))]
            is_new = 1
            for j in range(len(archive_xx)):
                x, f_x = archive_xx[j], archive_f[j]
                # Check whether share the same basin
                is_basin = True
                if (self.max_function_evaluations - self.n_function_evaluations) > self.max_budget:
                    is_basin = self.is_new_basin(y, x, f_y, f_x, args)
                if is_basin:  # share the same basin
                    if f_y < f_x:  # y is fitter than x
                        archive_xx[j], archive_f[j] = y, f_y
                    n_rep[i] += 1  # (Current restart)
                    is_new = 0  # It is not a new basin
                    break

            if is_new == 1:
                archive_xx = np.vstack((archive_xx, y))
                archive_f = np.hstack((archive_f, f_y))
                n_rep = np.hstack((n_rep, 0))  # (Current restart)
                archive_d = np.hstack((archive_d, hat_d_0))

        # Update the normalized taboo distance based on the (CURRENT) attraction power of the corresponding basin (2.4)
        diff = n_rep - self.alpha_new * (len(yy) / len(archive_xx))
        idx = np.power(-1, diff <= 0)  # A negative value indicates the basin has a weaker attraction power
        archive_d *= np.power(1 + abs(diff), self.tau_hat_d)
        for i in range(len(idx)):
            if idx[i] == -1:
                archive_d[i] = 1 / archive_d[i]  # Reduce the normalized distance
        return archive_xx, archive_f, archive_d

    def generate_subpopulation(self, mean, cov, sigma,  # current subpopulation mean, covariance, and sigma
                               superior_mean, superior_d,  # superior subpopulations; mean and normalized distance
                               archive_xx, archive_f, archive_d,  # all subpopulations; best
                               elt_xx, elt_f, elt_zz, elt_s,  # only current subpopulation; previous elites
                               best_f_ne, median_f_ne, args=None):  # only current subpopulation; history best, median
        """
        elt_n: Number of elite individuals per subpopulation
        elt_xx: initial elite solutions of the subpopulation
        elt_f: initial elite function fitness
        elt_zz: Initial value of the elite variation vector
        elt_s: Initial value of the elite global step sizes
        best_f_ne: History of the best of non-elite solutions for each subpopulation
        med_f_ne: History of the median of non-elite solutions for each subpopulation
        """
        temp_shrinkage_factor = 1  # for temporary shrinkage of the taboo regions
        values, vectors = np.linalg.eig(cov)
        vectors = np.real(vectors)
        values = np.abs(values)  # np.max(v, 0) ??? whether it is right in the author's matlab codes?
        inv_cov = np.matmul(np.matmul(vectors, np.diag(1 / values)), vectors.T)  # Inverse of the matrix
        sqrt_v = np.sqrt(values)

        # Generate taboo points
        taboo_points = np.copy(superior_mean)  # Center of fitter subpopulations
        taboo_points_d = np.copy(superior_d)  # The normalized taboo distance of fitter subpopulations
        for i in range(len(archive_xx)):  # Consider fitter archived points as taboo points
            aa = np.ones((elt_f.size,))[archive_f[i] < elt_f]
            if len(aa) == elt_f.size:  # if elt_f is empty, this will not continue
                taboo_points = np.append(taboo_points, archive_xx[i])
                taboo_points_d = np.append(taboo_points_d, archive_d[i])

        # Determine which taboo points are critical (numpy-ndarray)
        criticality_index = self.find_critical_points_descend_index(mean, sigma, inv_cov, taboo_points, taboo_points_d)

        # Sampling \lamda new solutions
        # the diversity preservation strategy is applied to the sampling stage
        xx = np.zeros((self.n_individuals, self.ndim_problem))
        f = np.zeros((self.n_individuals,))
        sigmas = np.zeros((self.n_individuals,))
        zz = np.copy(xx)
        # Generate \lambda taboo acceptable solutions
        for n in range(self.n_individuals):
            accept = False
            while not accept:
                sigmas[n] = sigma * np.exp(self.rng_optimization.normal(0, 1, 1) * self.tau)
                aa = sqrt_v * self.rng_optimization.normal(0, 1, (self.ndim_problem,))
                zz[n] = np.matmul(vectors, aa)
                xx[n] = mean + sigmas[n] * zz[n]
                if criticality_index.size == 0:
                    accept = True
                for c in criticality_index:
                    d = calculate_mahalanobis_distance(xx[n], taboo_points[c], inv_cov)
                    accept = d > sigma * taboo_points_d[c] * temp_shrinkage_factor
                    if not accept:  # reject
                        temp_shrinkage_factor *= self.c_red  # Temporary shrink the size of the taboo regions
                        break
            f[n] = self._evaluate_fitness(xx[n], args)

        # best for the non-elite solutions
        best_f_ne = np.append(best_f_ne, np.min(f))
        median_f_ne = np.append(median_f_ne, np.median(f))

        # Append the surviving elites from the previous generation (Recombine)
        # At the end of each generation, the solutions (the union of λ recently generated and Nelt surviving solutions
        # from the previous iteration) are sorted in increasing order of the function value.
        # Parameters of the subpopulations are then updated as follows:
        for k in range(len(elt_f)):
            accept = False
            for c in criticality_index:
                d = calculate_mahalanobis_distance(elt_xx[k], taboo_points[c], inv_cov)
                accept = d > sigma * taboo_points_d[c] * temp_shrinkage_factor
                if not accept:
                    break
            if accept:  # The elite was not in a taboo region
                xx = np.vstack((xx, elt_xx[k]))
                f = np.hstack((f, elt_f[k]))
                sigmas = np.hstack((sigmas, elt_s[k]))
                zz = np.vstack((zz, elt_zz[k]))

        # Update parameters of the subpopulation using Equation 6
        idx = np.argsort(f)
        new_mean = np.sum(self._w.reshape(self.n_parents, 1) * xx[idx[:self.n_parents]], 0) / self.n_parents
        n_d = self.ndim_problem
        c = np.sum(np.array([self._w[i] * np.tile(zz[idx[i]].reshape(n_d, 1), (1, n_d)) * np.tile(zz[idx[i]], (n_d, 1))
                             for i in range(self.n_parents)]))
        new_cov = (1 - 1 / self.tau_c) * cov + c / self.tau_c
        new_cov = (new_cov + new_cov.T) / 2  # A symmetric matrix
        numerator = np.exp(np.dot(self._w, np.log(sigmas[:self.n_parents])))
        denominator = np.exp(np.mean(np.log(sigmas)))
        new_sigma = sigma * numerator / denominator

        # Update the elite solutions
        elt_xx = xx[idx[:self.n_elt]]
        elt_f = f[idx[:self.n_elt]]
        elt_s = sigmas[idx[:self.n_elt]]
        elt_zz = zz[idx[:self.n_elt]]
        elt_xx_f_s_zz = (elt_xx, elt_f, elt_s, elt_zz)
        best_xx_f = (xx[idx[0]], f[idx[0]])
        new_mean_cov_sigma = (new_mean, new_cov, new_sigma)
        best_median_f_ne = (best_f_ne, median_f_ne)
        return best_xx_f, new_mean_cov_sigma, elt_xx_f_s_zz, best_median_f_ne

    # Algorithm 2 (2.7 Initialization of Subpopulations)
    def initialize_subpopulations(self, archive_xx, archive_d):
        hat_d = np.copy(self._hat_d_bak)
        if len(archive_d) > 0:
            # Set the default value of the normalized taboo distance based on the distribution of the current values
            # Equal to the $25\text{th}$ percentile of normalized taboo distances of the solutions stored in Archive.
            hat_d = np.percentile(archive_d, self.percentile)
        cov, sigma = np.diag(self.upper_boundary - self.lower_boundary), self._sigma_bak
        inv_cov = np.diag(np.power(self.upper_boundary - self.lower_boundary, -1))
        means = np.empty((self.n_s, self.ndim_problem))

        s, r = 0, 0
        while s < self.n_s:
            if r > 100:
                sigma, r = sigma + self.c_red * sigma, 0  # the size of each taboo region shrinks by (1-self.c_red)
            # Sample uniformly
            x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
            # for archived points
            accept = True
            for a in range(len(archive_xx)):
                d = calculate_mahalanobis_distance(archive_xx, x, inv_cov)
                if (2 * hat_d + archive_d[a]) * sigma > d:
                    accept = False
                    break
            # for previously sampled points
            if accept:
                for p in range(s):
                    if (3 * hat_d) * sigma > calculate_mahalanobis_distance(means[p], x, inv_cov):
                        accept = False
                        break
            if accept:
                means[s], r = x, 0
                s += 1
            else:
                r += 1

        # Defaults value of the normalized taboo distance (Use for the subpopulations in the current restart)
        means_d = hat_d * np.ones((len(means),))
        sigmas = sigma * np.ones((len(means),))
        return means, means_d, sigmas, cov

    def new_iterate(self, means, means_d, sigmas, cov, archive_xx, archive_f, archive_d, args=None):
        # for termination criterion
        n_no_improvement = 120 + math.floor(30 * self.ndim_problem / self.n_individuals)  # No improvement
        n_stalled_size = n_no_improvement - 110  # Stalled size (stagnation)

        # initial values
        elt_xx = {k: np.tile(means[k], (self.n_elt, 1)) for k in range(self.n_s)}
        elt_zz = {k: 0 * np.tile(means[k], (self.n_elt, 1)) for k in range(self.n_s)}
        elt_f = {k: np.ones((self.n_elt,)) * self.best_so_far_y for k in range(self.n_s)}
        elt_s = {k: np.ones((self.n_elt,)) * np.mean(sigmas) for k in range(self.n_s)}
        cov_dict = {k: cov for k in range(self.n_s)}
        best_f_ne = {k: np.arange(n_no_improvement)[::-1] for k in range(self.n_s)}
        median_f_ne = {k: np.arange(n_no_improvement)[::-1] for k in range(self.n_s)}
        superior_index = {k: np.arange(k) for k in range(self.n_s)}
        best_xx = np.copy(means)
        best_f = np.ones((self.n_s,)) * self.best_so_far_y_bak

        active_subpop = np.arange(self.n_s)  # Activating subpopulations
        terminate_subpop = np.empty((0,), dtype=int)  # Terminated subpopulations
        used_iteration = np.ones((self.n_s,))
        terminate_flag, n_iteration = 0, 0
        while (active_subpop.size >= 1) and (terminate_flag == 0):
            for m in active_subpop:
                # The number of potential global minima in the current restart. It is used only for termination of the
                # restart so that sufficient budget remains for analyzing the subpopulations later.
                # This is a specialization for GECCO2016 competition
                cnt = np.ones((len(best_f),))[(best_f - self.fitness_threshold) <= min(np.hstack((archive_f, best_f)))]
                cnt = 1 + np.sum(cnt)
                if (self.max_function_evaluations - self.n_function_evaluations) <= \
                        (cnt * self.max_budget * (len(archive_xx) + cnt / 2) + self.n_individuals):
                    self.n_function_evaluations += (self.max_function_evaluations - self.n_function_evaluations) + 1
                    terminate_flag = 1  # terminate the restart
                    break

                # Termination condition
                condition_number = np.linalg.cond(cov_dict[m])
                max_diff = max(best_f_ne[m][-n_stalled_size:]) - min(best_f_ne[m][-n_stalled_size:])
                # the median of the 20 newest values is not smaller than the median of the 20 oldest values,
                # excluding elites
                min_imp_best = np.median(best_f_ne[m][-20:])
                min_imp_best = min_imp_best - np.median(best_f_ne[m][-n_no_improvement:(-n_no_improvement + 20)])
                min_imp_med = np.median(median_f_ne[m][-20:])
                min_imp_med = min_imp_med - np.median(median_f_ne[m][-n_no_improvement:(-n_no_improvement + 20)])
                active_flags = [condition_number < self.n_condition, max_diff > self.fitness_threshold,
                                min_imp_best <= 0, min_imp_med <= 0, not self._check_terminations()]
                if not all(active_flags):  # Terminate the subpopulation if a stopping criterion is satisfied
                    means[m] = best_xx[m]
                    terminate_subpop = np.append(terminate_subpop, m)
                    continue

                used_iteration[m] = n_iteration + 1
                superior_mean = means[superior_index[m]]
                superior_d = means_d[superior_index[m]]
                best_xx_f, new_mean_cov_sigma, elt_xx_f_s_zz, best_median_f_ne = \
                    self.generate_subpopulation(
                        means[m], cov_dict[m], sigmas[m],
                        superior_mean, superior_d,
                        archive_xx, archive_f, archive_d,
                        elt_xx[m], elt_f[m], elt_zz[m], elt_s[m],
                        best_f_ne[m], median_f_ne[m], args)
                best_xx[m], best_f[m] = best_xx_f
                means[m], cov_dict[m], sigmas[m] = new_mean_cov_sigma
                elt_xx[m], elt_f[m], elt_s[m], elt_zz[m] = elt_xx_f_s_zz
                best_f_ne[m], median_f_ne[m] = best_median_f_ne

                # Find non-terminated subpopulations
            active_subpop = np.array([i for i in active_subpop if i not in terminate_subpop])
            if active_subpop.size != 0:
                active_subpop = active_subpop[np.argsort(best_f[active_subpop])]
                superior_index.update({m: np.arange(len(best_f))[best_f < best_f[m]] for m in active_subpop})
            n_iteration += 1

            # Update parameters (Stagnation:)
            n_no_improvement = 120 + math.floor(30 * self.ndim_problem / self.n_individuals + 0.2 * n_iteration)

            if self._check_terminations():
                break

        self.average_iteration_subpopulations = np.mean(used_iteration)
        return best_xx, best_f

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        ES.optimize(self, fitness_function)
        archive_xx, archive_f, archive_d = np.empty((0, self.ndim_problem)), np.empty((0,)), np.empty((0,))
        all_best_xx, all_best_f = {}, {}
        while True:
            # Initialize subpopulations (Algorithm 2)
            means, means_d, sigmas, cov = self.initialize_subpopulations(archive_xx, archive_d)

            # Iterate
            best_xx, best_f = self.new_iterate(means, means_d, sigmas, cov, archive_xx, archive_f, archive_d, args)
            all_best_xx.update({self._n_generations + 1: best_xx})
            all_best_f.update({self._n_generations + 1: best_f})

            # Update the archive using the best solutions of the converged subpopulations and the old archive
            # (Algorithm 1)
            archive_xx, archive_f, archive_d = self.updating_archive(best_xx, best_f, archive_xx, archive_f, archive_d)

            # Restart (Equations 8)
            self.update_population_size()

            if self._check_terminations():
                break
            self._n_generations += 1
        # return archive_xx, archive_f, archive_d, all_best_xx, all_best_f
        return archive_f
