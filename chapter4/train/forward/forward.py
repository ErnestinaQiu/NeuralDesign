""" forward part """
import numpy as np

def base_synapse_output(w, a):
    """
    the output of single synapse, ps w included b
    input
    we rule that every vector or matrix is vertical at first, that is, the num of the first dim is the same. We use mat.T if need
    @w(weights matrix, np.matrix) @a(input of this synapse, np.matrix)
    output
    @the input of cell(np.matrix)
    """
    out = np.matmul(w.T, a)
    return out


