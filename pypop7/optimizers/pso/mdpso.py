import numpy as np  # engine for numerical computing

from pypop7.optimizers.pso.pso import PSO  # abstract class of all particle swarm optimizer (PSO) classes


class MDPSO(PSO):
    """Multimodal Delayed Particle Swarm Optimizer (MDPSO).

    Following the velocity model of Song, Wang and Zou, the update adds two stochastic *delayed* terms
    built from randomly chosen past personal-best and global-best positions stored over previous
    generations. Concretely, at generation ``k`` the velocity update reads:

        v_i(k+1) = w*v_i(k)
                 + c1*r1*(p_i(k) - x_i(k))
                 + c2*r2*(p_g(k) - x_i(k))
                 + s_i*c1*r3*(p_i(k - tau_i) - x_i(k))
                 + s_g*c2*r4*(p_g(k - tau_g) - x_i(k))

    where ``tau_i``, ``tau_g`` are random integer delays drawn uniformly from ``[0, k]``, and the
    intensity factors ``s_i``, ``s_g`` are determined by the evolutionary state (swarm dispersion).
    Acceleration coefficients follow the PSO-TVAC schedule by default (Ratnaweera et al., 2004).

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
                * 'n_individuals' - swarm (population) size, aka number of particles (`int`, default: `20`),
                * 'cognition'     - cognitive learning rate when ``use_tvac`` is ``False`` (`float`, default: `2.0`),
                * 'society'       - social learning rate when ``use_tvac`` is ``False`` (`float`, default: `2.0`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`),
                * 'use_tvac'      - if ``True``, use time-varying acceleration coefficients (PSO-TVAC)
                                    as described in the paper, with ``c1`` decreasing from 2.5 to 0.5
                                    and ``c2`` increasing from 0.5 to 2.5 (`bool`, default: ``True``),
                * 'c1i'           - initial value of ``c1`` for PSO-TVAC (`float`, default: `2.5`),
                * 'c1f'           - final value of ``c1`` for PSO-TVAC (`float`, default: `0.5`),
                * 'c2i'           - initial value of ``c2`` for PSO-TVAC (`float`, default: `0.5`),
                * 'c2f'           - final value of ``c2`` for PSO-TVAC (`float`, default: `2.5`).

    References
    ----------
    Song, B., Wang, Z. and Zou, L., On Global Smooth Path Planning for Mobile Robots Using A Novel
    Multimodal Delayed PSO Algorithm. Brunel University Research Archive:
    https://bura.brunel.ac.uk/bitstream/2438/14201/1/FullText.pdf

    Kennedy, J. and Eberhart, R., 1995. Particle swarm optimization. IEEE ICNN.
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.use_tvac = options.get('use_tvac', True)
        self.c1i = options.get('c1i', 2.5)
        self.c1f = options.get('c1f', 0.5)
        self.c2i = options.get('c2i', 0.5)
        self.c2f = options.get('c2f', 2.5)
        self._pg_hist = []  # list of global-best position vectors, one per generation
        self._pp_hist = []  # list of (n_individuals, ndim_problem) personal-best snapshots, one per generation

    def initialize(self, args=None):
        v, x, y, p_x, p_y, n_x = PSO.initialize(self, args)
        self._pg_hist.clear()
        self._pp_hist.clear()
        g_idx = int(np.argmin(p_y))
        self._pg_hist.append(np.copy(p_x[g_idx]))
        self._pp_hist.append(np.copy(p_x))
        return v, x, y, p_x, p_y, n_x

    def _mean_distances(self, x):
        """Mean distance from each particle position to all other positions (eq. (6) structure)."""
        s = self.n_individuals
        if s <= 1:
            return np.zeros((s,))
        d = np.empty((s,))
        for i in range(s):
            idx = np.concatenate((np.arange(0, i), np.arange(i + 1, s)))
            d[i] = np.mean(np.linalg.norm(x[i] - x[idx], axis=1))
        return d

    def _evolutionary_factor(self, x, p_y):
        """Return Ef in [0, 1] (eq. (6)-(7)); degenerate spreads yield 0.5."""
        d = self._mean_distances(x)
        d_min, d_max = float(np.min(d)), float(np.max(d))
        g_idx = int(np.argmin(p_y))
        idx_others = np.concatenate((np.arange(0, g_idx), np.arange(g_idx + 1, self.n_individuals)))
        if len(idx_others) == 0:
            return 0.5
        d_g = float(np.mean(np.linalg.norm(x[g_idx] - x[idx_others], axis=1)))
        span = d_max - d_min
        if span <= 1e-30:
            return 0.5
        ef = (d_g - d_min) / span
        return float(np.clip(ef, 0.0, 1.0))

    @staticmethod
    def _state_from_ef(ef):
        """Map evolutionary factor to xi in {1,2,3,4} (eq. (8))."""
        if ef < 0.25:
            return 1
        if ef < 0.5:
            return 2
        if ef < 0.75:
            return 3
        return 4

    @staticmethod
    def _delay_intensity(xi, ef):
        """Table I: local/global delay strengths s_i, s_g."""
        if xi == 1:  # convergence: no delayed information
            return 0.0, 0.0
        if xi == 2:  # exploitation: only local delayed term
            return ef, 0.0
        if xi == 3:  # exploration: only global delayed term
            return 0.0, ef
        return ef, ef  # jumping-out: both delayed terms

    def _tvac_coefficients(self, k_iter):
        """PSO-TVAC c1, c2 (eq. (3)-(4)) with c1i=2.5, c2i=0.5, c1f=0.5, c2f=2.5."""
        t_max = max(float(self._max_generations), 1.0)
        t = float(k_iter)
        c1 = (self.c1i - self.c1f) * (t_max - t) / t_max + self.c1f
        c2 = (self.c2i - self.c2f) * (t_max - t) / t_max + self.c2f
        return c1, c2

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        # Velocity update (eq. 5):
        #   v_i += c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        #        + s_i*c1*r3*(pbest_i(k-tau_i) - x_i)
        #        + s_g*c2*r4*(gbest(k-tau_g) - x_i)
        # Delays tau_i, tau_g are random integers uniform on [0, k] (paper, Section III-A).
        k = self._n_generations
        ef = self._evolutionary_factor(x, p_y)
        xi = self._state_from_ef(ef)
        s_i, s_g = self._delay_intensity(xi, ef)

        if self.use_tvac:
            c1, c2 = self._tvac_coefficients(k)
        else:
            c1, c2 = self.cognition, self.society

        w = self._w[min(k, len(self._w) - 1)]

        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = p_x[np.argmin(p_y)]
            r1 = self.rng_optimization.uniform(size=(self.ndim_problem,))
            r2 = self.rng_optimization.uniform(size=(self.ndim_problem,))
            r3 = self.rng_optimization.uniform(size=(self.ndim_problem,))
            r4 = self.rng_optimization.uniform(size=(self.ndim_problem,))
            tau_i = int(self.rng_optimization.integers(0, k + 1))
            tau_g = int(self.rng_optimization.integers(0, k + 1))
            p_delayed = self._pp_hist[k - tau_i][i]
            g_delayed = self._pg_hist[k - tau_g]
            v[i] = (w*v[i] +
                    c1*r1*(p_x[i] - x[i]) +
                    c2*r2*(n_x[i] - x[i]) +
                    s_i*c1*r3*(p_delayed - x[i]) +
                    s_g*c2*r4*(g_delayed - x[i]))
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = x[i], y[i]

        g_idx = int(np.argmin(p_y))
        self._pg_hist.append(np.copy(p_x[g_idx]))
        self._pp_hist.append(np.copy(p_x))
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
