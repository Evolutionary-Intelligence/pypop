import ray  # for distributed computing
import numpy as np
from sklearn.preprocessing import Normalizer

import data_science as ds


@ray.remote
def ray_problem(r_w, r_x, r_y):  # to be shared across all nodes
    return ds.tanh_loss_lr(r_w, r_x, r_y)


if __name__ == '__main__':
    x, y = ds.read_qsar_androgen_receptor()
    transformer = Normalizer().fit(x)
    x = transformer.transform(x)
    ray_x, ray_y = ray.put(x), ray.put(y)
    w = 3.0*np.ones((x.shape[1] + 1,))
    ray_w = ray.put(w)
    ray_loss = ray_problem.remote(w, x, y)
    print(ray.get(ray_loss))  # 3.52815643380549
