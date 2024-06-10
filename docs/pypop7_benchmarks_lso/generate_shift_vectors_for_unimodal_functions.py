import numpy as np

import pypoplib.shifted_functions as cf
from pypoplib.shifted_functions import generate_shift_vector


def generate_shift_vectors(functions=None, ndims=None, seed=None):
    if functions is None:
        functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                     cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
    if ndims is None:
        ndims = [2, 10, 100, 200, 1000, 2000]
    if seed is None:
        seed = 20220501

    rng = np.random.default_rng(seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=(len(functions), len(ndims)))

    for i, f in enumerate(functions):
        for j, d in enumerate(ndims):
            generate_shift_vector(f, d, -9.5, 9.5, seeds[i, j])


if __name__ == '__main__':
    generate_shift_vectors()
