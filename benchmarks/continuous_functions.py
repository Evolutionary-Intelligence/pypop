import numpy as np

import base_functions
from benchmarks.shifted_functions import _load_shift_vector
from benchmarks.rotated_functions import _load_rotation_matrix


# helper functions
def _load_shift_and_rotation(func, x, shift_vector=None, rotation_matrix=None):
    shift_vector = _load_shift_vector(func, x, shift_vector)
    rotation_matrix = _load_rotation_matrix(func, x, rotation_matrix)
    return shift_vector, rotation_matrix


def sphere(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(sphere, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.sphere(x)
    return y


def cigar(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(cigar, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.cigar(x)
    return y


def discus(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = _load_shift_and_rotation(discus, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.discus(x)
    return y
