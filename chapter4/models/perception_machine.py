"""
date: 27th June 2021
author: Wenyu Qiu
description: some simple machines

"""
import numpy as np

class NotImplement(BaseException):
    def __init__(self, *args: object):
        super().__init__(*args)
    return 'Not implement yet'

# abstract class
class Model:
    """ an abstract class which can need to be impletement in the succession """
    def __init__(self, name):
        self.name = name
        self.construction = []
    
    # model construction
    ## layer
    def add(self, layer):
        """ add layer into one model, the layer can be modified according to class layer """
        self.construction.append(layer)
        return 
    
    ## predict
    def predict(self, input):
        if not self.construction:
            return 'please define the model before this step'
        y = input
        for layer in self.construction:
            y = layer(y)
        return y

    ## feedback
    def feedback(self):
        
            


    ## train
    def fit(self):
        """ the rule to train and update the weights """
        return NotImplement

class Input:
    def __init__(self, x):
        self.x = x
    


class Layer:
    def __init__(self, name=None):
        self.name = name

    def weights_matrix(self, width, height, bias, initiate_mode=0):
        if initiate_mode == 0:
            weights_matrix = np.zeros((width, weight))
    
    def 
        
