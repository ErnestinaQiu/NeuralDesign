"""
date: 20211029
author: Wenyu Qiu
des: sppy system
"""
import sympy as sp
import numpy as np
from sympy.core import expr
from diff import hessian, multi_diff

def steepest_descent(func, vars, start_p, lr):
    """ @func(only support square), @start_p(the start point), @lr(learning rate)"""
    if len(vars) != len(start_p):
        raise ValueError('len(vars) != len(start_p), len(vars)=={}, len(start_p)=={}'.format(len(vars), len(start_p)))
    gk = multi_diff(func, vars)
    subs_pairs = []
    for i in range(len(start_p)):
        subs_pairs.append((vars[i], start_p[i]))
    pk = gk.subs(subs_pairs)
    print(pk)
    end_p = start_p - pk*lr 
    return end_p

def cal_lr(func, vars):
    """ presume that the capacity func is square """
    hes_mat = hessian(func, vars)
    eigenvals = [val for val in hes_mat.eigenvals()]
    lr = round(1/max(eigenvals), 3)
    return lr

def show_implicity(func):
    """ intend to show the implicity function, but failed, don't know why """
    ezplot = lambda exper: sp.plot_implicit(sp.parse_expr(exper))
    ezplot(func)
    return None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x, y = sp.symbols('x, y')
    f = x**2+25*y**2
    # delta_f = sp.diff(f, x, y)
    # print(delta_f)
    # start_p = sp.Matrix([[0.5], [0.5]])
    # print(steepest_descent(f, [x, y], start_p, 0.01))
    lr = cal_lr(f, [x, y])
    print(lr);