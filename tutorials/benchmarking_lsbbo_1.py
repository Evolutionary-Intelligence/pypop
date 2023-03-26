import time

import numpy as np

from pypop7.benchmarks.shifted_functions import generate_shift_vector
from pypop7.benchmarks.rotated_functions import generate_rotation_matrix


def generate_sv_and_rm(functions=None, ndims=None, seed=None):
    if functions is None:
        functions = ['sphere', 'cigar', 'discus', 'cigar_discus', 'ellipsoid',
                     'different_powers', 'schwefel221', 'step', 'rosenbrock', 'schwefel12']
    if ndims is None:
        ndims = [2, 10, 100, 200, 1000, 2000]
    if seed is None:
        seed = 20221001

    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions), len(ndims)))

    for i, f in enumerate(functions):
        for j, d in enumerate(ndims):
            generate_shift_vector(f, d, -9.5, 9.5, seeds[i, j])

    start_run = time.time()
    for i, f in enumerate(functions):
        for j, d in enumerate(ndims):
            start_time = time.time()
            generate_rotation_matrix(f, d, seeds[i, j])
            print('* {:d}-d {:s}: runtime {:7.5e}'.format(
                d, f, time.time() - start_time))
    print('*** Total runtime: {:7.5e}.'.format(time.time() - start_run))


if __name__ == '__main__':
    generate_sv_and_rm()
