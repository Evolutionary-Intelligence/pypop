import time
import numpy as np

import pypoplib.rotated_functions as cf
from pypoplib.rotated_functions import generate_rotation_matrix


def generate_rotation_matrices(functions=None, ndims=None, seed=None):
    if functions is None:
        functions = [cf.griewank, cf.ackley, cf.rastrigin, cf.levy_montalvo, cf.michalewicz,
                     cf.salomon, cf.bohachevsky, cf.scaled_rastrigin, cf.skew_rastrigin, cf.schaffer]
    if ndims is None:
        ndims = [2, 10, 100, 200, 1000, 2000]
    if seed is None:
        seed = 20221002

    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions), len(ndims)))

    start_run = time.time()
    for i, f in enumerate(functions):
        for j, d in enumerate(ndims):
            start_time = time.time()
            generate_rotation_matrix(f, d, seeds[i, j])
            print('* {:d}-d {:s}: runtime {:7.5e}'.format(
                d, f.__name__, time.time() - start_time))
    print('*** Total runtime: {:7.5e}.'.format(time.time() - start_run))


if __name__ == '__main__':
    generate_rotation_matrices()
