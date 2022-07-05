"""Latent Action Monto Carlo Tree Search(LA-MCTS)
    Reference
    --------------
    L. Wang, R. Fonseca, Y. Tian
    Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search
    NeurIPS 2020
    https://proceedings.neurips.cc/paper/2020/hash/e2ce14e81dba66dbff9cbc35ecfdb704-Abstract.html
    Code of paper:
    https://github.com/facebookresearch/LaMCTS/tree/main/LA-MCTS
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
import copy as cp
from torch.quasirandom import SobolEngine

from pypop7.optimizers.core.optimizer import Optimizer


# generate initial points
def latin_hyper_cube(n, dims, ub, lb):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n))
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]
    perturbation = np.random.uniform(-1.0, 1.0, (n, dims))
    perturbation = perturbation / float(2 * n)
    points += perturbation
    points = points * (ub - lb) + lb
    return points


# Use for classifying points
class Classifier:
    def __init__(self, samples, dimension, kernel_type, gamma_type):
        self.dimension = dimension
        # create gaussian regressor
        noise = 0.1
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)
        self.kmeans = KMeans(n_clusters=2)
        # data structure
        self.samples = []
        self.x = []
        self.fx = []
        # learn boundary
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type
        self.svm = SVC(kernel=kernel_type, gamma=gamma_type)
        self.update_samples(samples)

    def get_mean(self):
        if len(self.fx) == 0:
            return float('inf')
        else:
            return np.mean(self.fx)

    def update_samples(self, samples):
        self.x = []
        self.fx = []
        self.samples = samples
        for sample in samples:
            self.x.append(sample[0])
            self.fx.append(sample[1])
        self.x = np.asarray(self.x, dtype=np.float64).reshape(-1, self.dimension)
        self.fx = np.asarray(self.fx, dtype=np.float64).reshape(-1)

    def learn_cluster(self):
        data = np.concatenate((self.x, self.fx.reshape([-1, 1])), axis=1)
        self.kmeans = self.kmeans.fit(data)
        plabel = self.kmeans.predict(data)
        zero_label_fx = []
        one_label_fx = []
        for i in range(len(plabel)):
            if plabel[i] == 0:
                zero_label_fx.append(self.fx[i])
            elif plabel[i] == 1:
                one_label_fx.append(self.fx[i])
            else:
                print("There should only be two cluster")
        zero_label_mean = np.mean(zero_label_fx)
        one_label_mean = np.mean(one_label_fx)
        if zero_label_mean > one_label_mean:
            for i in range(len(plabel)):
                if plabel[i] == 0:
                    plabel[1] = 1
                else:
                    plabel[i] = 0
        return plabel

    def expected_improvement(self, x, xi=0.001):
        x_sample = self.x
        mu, std = self.gpr.predict(x, return_std=True)
        mu_sample = self.gpr.predict(x_sample)
        y_max = np.max(mu_sample)
        a = (mu - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    def get_sample_ratio_in_region(self, x, path):
        length = len(x)
        for node in path:
            if len(x) == 0:
                return 0, np.array([])
            boundary = node[0].classifier.svm
            x = x[boundary.predict(x) == node[1]]
        ratio = len(x) / length
        return ratio, x

    def split_data(self):
        good_samples = []
        bad_samples = []
        if len(self.samples) == 0:
            return good_samples, bad_samples
        plabel = self.learn_cluster()
        assert len(plabel) == len(self.x)
        if (1 in plabel) == True:
            self.svm.fit(self.x, plabel)
            for i in range(len(plabel)):
                if plabel[i] == 0:
                    good_samples.append(self.samples[i])
                else:
                    bad_samples.append((self.samples[i]))
            assert len(good_samples) + len(bad_samples) == len(self.samples)
            return good_samples, bad_samples, True
        else:
            return self.samples, [], False

    def propose_rand_sample_sobol(self, sample_num, path, lower_boundary, upper_boundary):
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dimension, scramble=True, seed=seed)

        ratio_check, centers = self.get_sample_ratio_in_region(self.x, path)
        if ratio_check == 0 or len(centers) == 0:
            return np.random.uniform(lower_boundary, upper_boundary, size=(sample_num, self.dimension))

        final_cands = []
        for center in centers:
            center = self.x[np.random.randint(len(self.x))]
            cands = sobol.draw(2000).to(dtype=torch.float64).cpu().detach().numpy()
            ratio = 1
            leng = 0.0001
            Blimit = np.max(upper_boundary - lower_boundary)
            while ratio == 1 and leng < Blimit:
                lb = np.clip(center - leng/2, lower_boundary, upper_boundary)
                ub = np.clip(center + leng/2, lower_boundary, upper_boundary)
                cands_ = cp.deepcopy(cands)
                cands_ = (ub - lb) * cands_ + lb
                ratio, cands_ = self.get_sample_ratio_in_region(cands_, path)
                if ratio < 1:
                    final_cands.extend(cands_.tolist())
                leng *= 2
        final_cands = np.array(final_cands)
        if len(final_cands) > sample_num:
            final_cands_idx = np.random.choice(len(final_cands), sample_num)
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return np.random.uniform(lower_boundary, upper_boundary, size=(sample_num, self.dimension))
            else:
                return final_cands

    def propose_sample_bo(self, sample_num, path, lower_bound, upper_bound, samples):
        # train the gpr
        x = []
        fx = []
        for sample in samples:
            x.append(sample[0])
            fx.append(sample[1])
        x = np.asarray(x).reshape(-1, self.dimension)
        fx = np.asarray(fx).reshape(-1)
        self.gpr.fit(x, fx)

        if len(path) == 0:
            return np.random.uniform(lower_bound, upper_bound, size=(sample_num, self.dimension))
        nums_rand_samples = 10000
        x = self.propose_rand_sample_sobol(nums_rand_samples, path, lower_bound, upper_bound)
        if len(x) == 0:
            return np.random.uniform(lower_bound, upper_bound, size=(sample_num, self.dimension))
        # select x with least expect improvement
        expect_improves = self.expected_improvement(x, xi=0.001)
        order = np.argsort(expect_improves)[:sample_num]
        return x[order]


# Use for build tree
class Node:
    obj_counter = 0
    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.

    def __init__(self, parent=None, dimension=10, is_reset=False, kernel_type='rbf', gamma_type='auto'):
        self.dimension = dimension
        self.parent = parent
        self.kids = []  # 0:good, 1:bad
        self.n = 0  # represent number of visit this node
        self.value = float('inf')  # node value equal to mean of object values in objects set
        self.objects = []
        self.is_svm_splittable = False
        self.classifier = Classifier([], self.dimension, kernel_type, gamma_type)
        if is_reset:
            Node.obj_counter = 0
        self.id = Node.obj_counter
        Node.obj_counter += 1

    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append(good_kid)
        self.kids.append(bad_kid)
        assert self.kids[0].classifier.get_mean() < self.kids[1].classifier.get_mean()

    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False

    def train_and_split(self):
        self.classifier.update_samples(self.objects)
        good_kid_data, bad_kid_data, is_split = self.classifier.split_data()
        return good_kid_data, bad_kid_data, is_split

    def update_objects(self, samples):
        self.objects.clear()
        self.objects.extend(samples)
        self.classifier.update_samples(self.objects)
        if len(self.objects) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = True
        self.value = self.classifier.get_mean()
        self.n = len(self.objects)

    def propose_sample_bo(self, sample_num, path, lower_boundary, upper_boundary, samples):
        propose_x = self.classifier.propose_sample_bo(sample_num, path, lower_boundary,
                                                      upper_boundary, samples)
        return propose_x

    def clear_data(self):
        self.objects.clear()

    def visit(self):
        self.n += 1

    def get_ucb(self, cp):
        if self.n == 0 or self.parent is None:
            return float('inf')
        else:
            # according to the paper
            # return self.value / self.n + 2 * Cp * np.sqrt(2 * np.log(self.parent.n) / self.n)
            # according to the code
            return self.value + 2 * cp * np.sqrt(2 * np .power(self.parent.n, 0.5) / self.n)

    def get_name(self):
        return "node" + str(self.id)

    def get_value(self):
        return self.value


class LAMCTS(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.kernel_type = options.get('kernel_type')
        self.gamma_type = options.get('gamma_type')
        self.Cp = options.get('Cp')
        self.leaf_size = options.get('leaf_size')
        self.solver_type = options.get('solver_type')
        self._n_generations = 0
        self.nodes = []
        self.samples = []
        self.sample_counter = 0
        self.current_best_sample = None
        self.current_best_value = float('inf')

        # initialize the root
        root = Node(parent=None, dimension=self.ndim_problem,
                    is_reset=True, kernel_type=self.kernel_type, gamma_type=self.gamma_type)
        self.nodes.append(root)
        self.ROOT = root

    def initialize(self, is_restart=False):
        values = []
        points = latin_hyper_cube(self.n_individuals, self.ndim_problem,
                                  self.upper_boundary, self.lower_boundary)
        for point in points:
            values.append(self.collect_samples(point))
        return values

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() is True and len(node.objects) > self.leaf_size and node.is_svm_splittable is True:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)

    def dynamic_treeify(self):
        # clean the original tree
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()

        # reset the root
        new_root = Node(parent=None, dimension=self.ndim_problem, is_reset=True,
                        kernel_type=self.kernel_type, gamma_type=self.gamma_type)
        self.nodes.append(new_root)
        self.ROOT = new_root
        self.ROOT.update_objects(self.samples)

        # build the tree
        while True:
            status = self.get_leaf_status()
            if True in status:
                split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
                for idx in split_by_samples:
                    parent = self.nodes[idx]
                    good_kid_data, bad_kid_data, is_split = parent.train_and_split()
                    if is_split is True:
                        good_kid = Node(parent, self.ndim_problem, False, self.kernel_type, self.gamma_type)
                        bad_kid = Node(parent, self.ndim_problem, False, self.kernel_type, self.gamma_type)
                        good_kid.update_objects(good_kid_data)
                        bad_kid.update_objects(bad_kid_data)
                        if good_kid.classifier.get_mean() < bad_kid.classifier.get_mean():
                            parent.update_kids(good_kid, bad_kid)
                            self.nodes.append(good_kid)
                            self.nodes.append(bad_kid)
                        else:
                            parent.update_kids(bad_kid, good_kid)
                            self.nodes.append(bad_kid)
                            self.nodes.append(good_kid)
                    else:
                        self.nodes[idx].is_svm_splittable = False
            else:
                break

    def collect_samples(self, x, value=None):
        if value is None:
            value = self._evaluate_fitness(x)
        if value < self.current_best_value:
            self.current_best_value = value
            self.current_best_sample = x
        self.sample_counter += 1
        self.samples.append((x, value))
        return value

    def select(self):
        current_node = self.ROOT
        path = []
        while current_node.is_leaf() is False:
            uct = []
            for kid in current_node.kids:
                uct.append(kid.get_ucb(self.Cp))
            choice = np.random.choice(np.argwhere(uct == np.amin(uct)).reshape(-1), 1)[0]
            path.append((current_node, choice))
            current_node = current_node.kids[choice]
        return current_node, path

    # refresh the value and num of a line of tree
    def back_propagate(self, leaf, value):
        current_node = leaf
        while current_node is not None:
            current_node.value = (current_node.value * current_node.n + value) / (current_node.value + 1)
            current_node.n += 1
            current_node = current_node.parent

    def iterate(self, leaf, path):
        values = []
        if self.solver_type == 'bo':
            samples = leaf.propose_sample_bo(1, path, self.lower_boundary,
                                             self.upper_boundary, self.samples)
        else:
            raise Exception("Solver not implemented")
        for i in range(len(samples)):
            if self.solver_type == 'bo':
                value = self.collect_samples(samples[i])
            else:
                raise Exception("Solver not implemented")
            self.back_propagate(leaf, value)
            values.append(value)
        return values

    def optimize(self, fitness_function=None):
        fitness = Optimizer.optimize(self, fitness_function)
        values = self.initialize()
        if self.record_fitness:
            fitness.extend(values)
        while True:
            self.dynamic_treeify()
            leaf, path = self.select()
            values = self.iterate(leaf, path)
            if self.record_fitness:
                fitness.extend(values)
            if self._check_terminations():
                break
            self._n_generations += 1
            self._print_verbose_info(values)
        results = self._collect_results(fitness)
        return results

    def _print_verbose_info(self, y):
        if self.verbose and (not self._n_generations % self.verbose_frequency):
            best_so_far_y = -self.best_so_far_y if self._is_maximization else self.best_so_far_y
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect_results(self, fitness):
        results = Optimizer._collect_results(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
