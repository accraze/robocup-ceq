import numpy as np


def array_equals(a, b):
    return np.array_equal(a, b)


def lookup_vector_index(space, vector):
    idx = None
    for i in range(space.shape[0]):
        if array_equals(space[i], np.array(vector)):
            idx = i
            break
    return idx
