""" for some simple feedback tactic """
import numpy as np

def base_update(w, e, z):
    """ only shift and translate in ndim Euclid space
    @w(general weights matrix, including bias) @e(the dir and stride of improvement) @z(general input of this synapse including 1 and p) """
    update_quantity = np.multiply(e, z)
    new_w = w + update_quantity
    return new_w







##############################################################
register = {'base_update': base_update}

def feedback(option, w, e, z):
    feed_func = register[option]
    new_w = feed_func(w, e, z)
    return new_w 
    