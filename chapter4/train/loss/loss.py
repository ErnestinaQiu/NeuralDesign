""" loss func """
import numpy as np

def euclid_loss(y_real, y_pred):
    loss = np.abs(y_real-y_pred)
    return loss

def square_loss(y_real, y_pred):
    loss = np.sqrt(np.power(y_real-y_pred, 2))
    return loss
    

