import numpy as np

from pypop7.optimizers.ds.ds import DS


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def put(self, data):
        self.queue.append(data)

    # for popping an element based on Priority
    def delete(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()


def basis(i, n):
    return np.identity(n)[:, i]


def peek(stack):
    if stack:
        return stack.queue[-1]
    else:
        return None


class Interval:
    def __init__(self, c, y, depths):
        self.c = c
        self.y = y
        self.depths = depths


def reparameterize_to_unit_hypercube(fun, a, b):
    delta = np.subtract(b, a)
    return lambda x: fun(np.multiply(x, delta) + a)


def rev_unit_hypercube_parameterization(x, a, b):
    return np.multiply(x, b - a) + a


def vertex_dist(interval):
    x = 0.5 * 3.0 ** (-interval.depths)
    return np.linalg.norm(x, ord=2)


def min_depth(interval):
    return np.min(interval.depths)


def divide(g, interval):
    c, d, n = np.array(interval.c), min_depth(interval), len(interval.c)
    dirs = np.asarray(np.where(interval.depths == d))[0]
    cs = [(c + np.multiply(3.0 ** (-d - 1), basis(i, n)), c - np.multiply((3.0 ** (-d - 1)), basis(i, n))) for i in
          dirs]
    vs = [(g(C[0]), g(C[1])) for C in cs]
    minvals = [np.min(V) for V in vs]
    intervals = []
    depths = np.copy(interval.depths)
    for j in np.argsort(minvals):
        depths[dirs[j]] += 1
        C, V = cs[j], vs[j]
        intervals.append(Interval(C[0], V[0], np.copy(depths)))
        intervals.append(Interval(C[1], V[1], np.copy(depths)))
    intervals.append(Interval(c, interval.y, np.copy(depths)))
    return intervals


def slope(y, y1, x, x1):
    if x == x1:
        x1 += 1e-10
    return (y1 - y) / (x1 - x)


def get_opt_intervals(intervals, eps):
    stack = []
    for pq in intervals:
        if len(intervals[pq]) != 0:
            interval = peek(intervals[pq])[0]
            x, y = vertex_dist(interval), interval.y
            while len(stack) > 1:
                interval1, interval2 = stack[-1], stack[-2]
                x1, y1 = vertex_dist(interval1), interval1.y
                x2, y2 = vertex_dist(interval2), interval2.y
                l = slope(y, y2, x, x2)
                if y1 <= l * (x1 - x) + y + eps:
                    break
                stack.pop()
            if len(stack) != 0 and y > stack[-1].y + eps:
                continue
            stack.append(interval)
    return stack


class Direct(DS):
    """Divided Rectangles(DIRECT)
        Reference
        ------------
        Kochenderfer, M.J. and Wheeler, T.A., 2019.
        Algorithms for optimization.
        MIT Press.
        https://algorithmsbook.com/optimization/files/chapter-7.pdf
        (See Algorithm 7.8 (Page 117) for details.)
        The original code can be viewed in
        https://github.com/ch4ki/DividedRectangles
        D. R. Jones, C. D. Perttunen, B. E. Stuckman
        A dividing rectangles algorithm for stochastic simulation optimization.
        JOURNAL OF OPTIMIZATION THEORY AND APPLICATION: Vol. 79, No. 1, OCTOBER 1993
        https://link.springer.com/article/10.1007/BF00941892
    """
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self.eps = options.get('eps')
        self.x = None

    def initialize(self):
        g = reparameterize_to_unit_hypercube(self.fitness_function, self.lower_boundary, self.upper_boundary)
        intervals = {}
        # x = self._initialize_x()
        # c = (x - self.lower_boundary) / (self.upper_boundary - self.lower_boundary)
        c = 0.5 * np.ones(self.ndim_problem)
        filling = np.empty(self.ndim_problem)
        filling.fill(0)
        interval = Interval(c, g(c), filling)
        self.add_interval(intervals, interval)
        return g, intervals

    def add_interval(self, intervals, interval):
        d = vertex_dist(interval)
        if d in intervals:
            pass
        else:
            intervals[d] = PriorityQueue()
        intervals[d].put((interval, interval.y))
        self.n_function_evaluations += 1
        if (not self._is_maximization) and (interval.y < self.best_so_far_y):
            self.best_so_far_x, self.best_so_far_y = \
                rev_unit_hypercube_parameterization(interval.c, self.lower_boundary, self.upper_boundary), interval.y
        if self._is_maximization and (-interval.y > self.best_so_far_y):
            self.best_so_far_x, self.best_so_far_y = \
                rev_unit_hypercube_parameterization(interval.c, self.lower_boundary, self.upper_boundary), -interval.y

    def iterate(self, g, intervals):
        s = get_opt_intervals(intervals, self.eps)
        to_add, y = [], []
        for interval in s:
            to_add.append(divide(g, interval))
            del intervals[vertex_dist(interval)]
        for add in to_add:
            for interval in add:
                self.add_interval(intervals, interval)
                y.append(interval.y)
        return intervals, y

    def optimize(self, fitness_function=None):
        fitness = DS.optimize(self, fitness_function)
        g, intervals = self.initialize()
        fitness.append(self.best_so_far_y)
        while True:
            if self._check_terminations():
                break
            intervals, y = self.iterate(g, intervals)
            if self.record_fitness:
                fitness.extend(y)
            self._n_generations += 1
            self._print_verbose_info(y)
        results = self._collect_results(fitness)
        return results
