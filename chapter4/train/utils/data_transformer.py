""" some tool for data reconstruct and convert """
import numpy as np

def trans_matrix(head, tail):
    ''' merge matrix into one according to rule '''
    if head.shape[1] != tail.shape[1]:
        raise ValueError('head.shape[1]({}) != tail.shape[1]({})'.format(head.shape[1], tail.shape[1]))
    if tail.shape[0] != 1:
        raise ValueError('tail.shape[0]({}) != 1'.format(tail.shape[0])) 
    new_matrix = np.zeros((head.shape[0]+tail.shape[0], head.shape[1]))
    new_matrix[:, :head.shape[1]] = head
    new_matrix[:, head.shape[1]] = tail
    return new_matrix