"""
date: 20211029
author: Wenyu Qiu
des: 1. Employ sppy system  2. We define matrix with shape as 2*1 as [[1], [2]] 
"""
import sympy as sp
import numpy as np
from sympy.core import expr
from diff import hessian, multi_diff

def steepest_descent(func, vars, start_p, lr):
    """ @func(only support square), @vars, @start_p(the start point), @lr(learning rate)"""
    if len(vars) != len(start_p):
        raise ValueError('len(vars) != len(start_p), len(vars)=={}, len(start_p)=={}'.format(len(vars), len(start_p)))
    gk = multi_diff(func, vars)
    subs_pairs = []
    for i in range(len(start_p)):
        subs_pairs.append((vars[i], start_p[i]))
    pk = gk.subs(subs_pairs)
    print('pk: {}'.format(pk))
    end_p = start_p - pk*lr
    return end_p

def newton_descent(func, vars, start_p, lr):
    """ @func(only support square), @vars, @start_p(the start point), @lr(learning rate) """
    if len(vars) != len(start_p):
        raise ValueError('len(vars) != len(start_p), len(vars)=={}, len(start_p)=={}'.format(len(vars), len(start_p)))
    gk = multi_diff(func, vars)
    subs_pairs = []
    for i in range(len(start_p)):
        subs_pairs.append((vars[i], start_p[i]))
    

def cal_lr(func, vars):
    """ presume that the capacity func is square, @func, @vars """
    hes_mat = hessian(func, vars)
    eigenvals = [val for val in hes_mat.eigenvals()]
    lr = 2*round(1/max(eigenvals), 3)
    return lr

def cal_lr_straight_line(func, vars, start_p):
    """ @func, @vars(tuple, or list, or array), @start_p(tuple, or list, or array) 
    eg: vars=(x, y), start_p=(1, 5) 
    """
    _gk = multi_diff(func, vars)
    subs_pairs = []
    for i in range(len(vars)):
        subs_pairs.append((vars[i], start_p[i]))
    gk = _gk.subs(subs_pairs)  #
    pk = -gk
    Ak = hessian(func, vars).subs(subs_pairs)
    lr_numerator = gk.T*pk
    _lr_denominator = pk.T*Ak*pk
    lr_denomonator = _lr_denominator.subs(subs_pairs)
    lr = -round(np.sum(lr_numerator)/np.sum(lr_denomonator), 4)
    return lr

def show_implicity(func):
    """ intend to show the implicity function, but failed, don't know why """
    ezplot = lambda exper: sp.plot_implicit(sp.parse_expr(exper))
    ezplot(func)
    return None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # deepest_descent
    # x, y = sp.symbols('x, y')
    # f = x**2+25*y**2
    # delta_f = sp.diff(f, x, y)
    # print(delta_f)
    # start_p = sp.Matrix([[0.5], [0.5]])
    # print(steepest_descent(f, [x, y], start_p, 0.01))
    ## cal_lr
    # lr = cal_lr(f, [x, y], mode='straight_line')
    # print(lr)
    ## cal_lr_straight
    x, y = sp.symbols('x, y')
    vars = [x, y]
    vec = sp.Matrix([[x], [y]])
    A = sp.Matrix([[2, 1], [1, 2]])
    f = 0.5*vec.T*A*vec
    start_p = sp.Matrix([[0.8], [-0.25]])
    lr = cal_lr_straight_line(f, [x, y], start_p=start_p)
    print(lr)
    subs_pairs = []
    for i in range(len(vars)):
        subs_pairs.append((vars[i], start_p[i]))
    _gk = multi_diff(f, [x, y])
    gk = _gk.subs(subs_pairs)
    next_p = start_p - gk*lr
    print(next_p)
    
    