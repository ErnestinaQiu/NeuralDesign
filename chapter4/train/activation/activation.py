""" 
date: 28th June 2021
author: Wenyu Qiu
des: some simple choices for activation 
"""
import numpy as np

def hardlim(a):
    """ 'a' means the output of the synapsis """
    if a > 0:
        return 1
    else:
        return 0

def hardlims(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    elif a == 0:
        return 0
    
register = {'hardlim': hardlim, 'hardlims': hardlims}



