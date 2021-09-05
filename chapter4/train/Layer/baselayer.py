""" basic methods of construction of layer """
import numpy as np

class Layer:
    def __init__(self, name=None):
        self.name = name

    def weights_matrix(self, width, height, bias, initiate_mode=0):
        if initiate_mode == 0:
            weights_matrix = np.zeros((width, weight))
    
