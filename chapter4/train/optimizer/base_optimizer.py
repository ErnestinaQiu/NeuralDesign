""" some basic optimizer """

def linear_optimizer(y_pred, y_real):
    """ only use residual to determine the dir of improvement """
    e = y_real - y_pred
    return e

