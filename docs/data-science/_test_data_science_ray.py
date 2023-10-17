import time

import ray  # for distributed computing
import numpy as np
from sklearn.preprocessing import Normalizer

import pypop7.benchmarks.data_science as ds


@ray.remote
def ray_tanh_loss_lr(r_w, r_x, r_y):  # to be shared across all nodes
    return ds.tanh_loss_lr(r_w, r_x, r_y)


@ray.remote
class Optimizer:
    def __init__(self, ndim_problem):
        self.ndim_problem = ndim_problem

    def optimize(self, fitness_function, r_w, r_x, r_y):
        assert self.ndim_problem == len(r_w)
        assert self.ndim_problem == r_x.shape[1] + 1
        assert r_x.shape[0] == len(r_y)
        return fitness_function(r_w, r_x, r_y)


@ray.remote
def ray_cross_entropy_loss_lr(r_w, r_x, r_y):  # to be shared across all nodes
    return ds.cross_entropy_loss_lr(r_w, r_x, r_y)


if __name__ == '__main__':
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    ray_x, ray_y = ray.put(x), ray.put(y)
    w = 3.0*np.ones((x.shape[1] + 1,))
    ray_w = ray.put(w)
    ray_loss = ray_tanh_loss_lr.remote(w, x, y)
    print(ray.get(ray_loss))  # 3.52815643380549
    optimizer = Optimizer.remote(len(w))
    print(ray.get(optimizer.optimize.remote(ds.tanh_loss_lr, w, x, y)))  # 3.52815643380549

    n_trials = 10_000
    x, y = ds.read_parkinson_disease_classification()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ds.cross_entropy_loss_lr(w, x, y))
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # * Runtime:  78.31092715263367  - Sum:  17379.784430717365

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ray_cross_entropy_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # * Runtime:  16.65428137779236  - Sum:  17379.784430717365

    start_time = time.time()
    s, d = 0.0, []
    w, x, y = ray.put(w), ray.put(x), ray.put(y)  # to be shared
    for i in range(n_trials):
        d.append(ray_cross_entropy_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # * Runtime:  4.1619696617126465  - Sum:  17379.784430717365
