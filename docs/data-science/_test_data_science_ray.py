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


@ray.remote
def ray_square_loss_lr(r_w, r_x, r_y):  # to be shared across all nodes
    return ds.square_loss_lr(r_w, r_x, r_y)


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

    n_trials = 100_000
    x, y = ds.read_parkinson_disease_classification()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    w = 3.0*np.ones((x.shape[1] + 1,))

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ds.cross_entropy_loss_lr(w, x, y))
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 423.94107270240784 - Sum:  173797.8443071736

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ray_cross_entropy_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 184.14105248451233 - Sum:  173797.8443071736

    start_time = time.time()
    s, d = 0.0, []
    w, x, y = ray.put(w), ray.put(x), ray.put(y)  # to be shared
    for i in range(n_trials):
        d.append(ray_cross_entropy_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 30.191553831100464 - Sum:  173797.8443071736

    x, y = ds.read_qsar_androgen_receptor(is_10=True)
    w = 3.0*np.ones((x.shape[1] + 1,))

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ds.square_loss_lr(w, x, y))
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 721.3659870624542 - Sum:  88203.91227030235

    start_time = time.time()
    s, d = 0.0, []
    for i in range(n_trials):
        d.append(ray_square_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 1497.7815556526184 - Sum:  88203.91227030235

    start_time = time.time()
    s, d = 0.0, []
    w, x, y = ray.put(w), ray.put(x), ray.put(y)  # to be shared
    for i in range(n_trials):
        d.append(ray_square_loss_lr.remote(w, x, y))
    d = ray.get(d)
    print('* Runtime: ', time.time() - start_time, ' - Sum: ', np.sum(d))
    # *Runtime: 124.87141919136047 - Sum:  88203.91227030235
