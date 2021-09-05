"""
date: 27th June 2021
author: Wenyu Qiu
description: some simple machines

"""
import numpy as np
from copy import deepcopy
from train.callback.simple_callback import feedback
from train.forward.forward import base_synapse_output
from train.optimizer.base_optimizer import linear_optimizer


class NotImplement(BaseException):
    def __init__(self, *args: object):
        super().__init__(*args)
        return ''.join([args, 'not implement yet'])

# abstract class
class Model:
    """ an abstract class which can need to be implement in the succession """
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
    def feedback(self, w, e, z, option='base_update'):
        """ the update engine, attention w and z are both general params
        @w(includes bias) @e(the dir of refinement) @z(the input of input, including p and 1) """
        new_w = feedback(option, w, e, z)
        return new_w
    
    ## the dir of refinement
    def improve_sign(self, y_pred, y_real):
        e = linear_optimizer(y_pred, y_real)
        return e
    
    ## show the construction of model
    def summary(self):
        return NotImplement
    
    ## choose the metrics to optimizer, 
    def compile(self, loss, optimizer, metrics):
        self.loss = loss
        return

    ## train
    def fit(self, x, y, stop_thresh_dict):
        """ the rule to train and update the weights """
        tmp = deepcopy(x)
        for layer in self.construction:
            tmp = layer(x)
        
        return NotImplement


    



        
