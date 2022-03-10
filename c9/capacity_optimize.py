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

def newton_descent(func, vars, start_p):
    """ only support square func @func(only support square), @vars, @start_p(the start point) """
    if len(vars) != len(start_p):
        raise ValueError('len(vars) != len(start_p), len(vars)=={}, len(start_p)=={}'.format(len(vars), len(start_p)))
    _gk = multi_diff(func, vars)
    _Ak = hessian(func, vars)
    subs_pairs = []
    for i in range(len(start_p)):
        subs_pairs.append((vars[i], start_p[i]))
    gk = _gk.subs(subs_pairs)
    Ak = _Ak.subs(subs_pairs)
    print('gk: {}, Ak: {}'.format(gk, Ak))
    delta_xk = Ak.inv()*gk
    end_p = start_p - delta_xk
    return end_p

def cal_lr(func, vars):
    """ presume that the capacity func is square, @func, @vars """
    hes_mat = hessian(func, vars)
    eigenvals = [val for val in hes_mat.eigenvals()]
    lr = 2*round(1/max(eigenvals), 3)
    return lr

def cal_lr_straight_line(func, vars, start_p, A=None, debug=False):
    """ @func, @vars(tuple, or list, or array), @start_p(tuple, or list, or array) 
    eg: vars=(x, y), start_p=(1, 5)
    """
    _gk = multi_diff(func, vars)
    subs_pairs = []
    for i in range(len(vars)):
        subs_pairs.append((vars[i], start_p[i]))
    gk = _gk.subs(subs_pairs)  #
    pk = -1*gk
    if not A:
        Ak = hessian(func, vars).subs(subs_pairs)
    else:
        Ak = A
    lr_numerator = gk.T*pk
    _lr_denominator = pk.T*Ak*pk
    lr_denomonator = _lr_denominator.subs(subs_pairs)
    if debug:
        print('gk.T: {}, pk: {}'.format(gk.T, pk))
        print('pk.T: {}, Ak: {}, pk: {}'.format(pk.T, Ak, pk))
    lr = -round(np.sum(lr_numerator)/np.sum(lr_denomonator), 4)
    return lr

def show_implicity(func):
    """ intend to show the implicity function, but failed, don't know why """
    ezplot = lambda exper: sp.plot_implicit(sp.parse_expr(exper))
    ezplot(func)
    return None

def conjungate_gradient_descent(func, vars, start_p, epsilon=0.001, max_turns=1000, beta_mode='F_R', A=None, debug=True):
    """ @func, only support square func, @vars, @start_p, @beta_mode, concludes 'H_S', 'F_R', 'P_R' 
    lr use the def cal_lr_straight_line
    """
    _gk = multi_diff(func, vars)
    if not A:
        A = hessian(func, vars)  # there will be no vars inside because it is square func
    
    # g0
    subs_pairs_0 = get_subs_pairs(vars, start_p)
    g0 = _gk.subs(subs_pairs_0)
    del subs_pairs_0
    lr_0 = cal_lr_straight_line(func=func, vars=vars, start_p=start_p, A=A, debug=debug)
    print('lr_0: {}'.format(lr_0))
    p0 = -1*g0
    if debug:
        print('p0: {}'.format(p0))
    x_last = None
    x_cur = start_p
    x_next = x_cur + lr_0*p0
    if debug:
        print('x_next: {}'.format(x_next))
    pk_cur = p0
    pk_last = None
    lt_cur = lr_0
    gk_cur = g0
    lt_last = None
    count=0
    while (abs(x_next[0] - x_cur[0]) >= epsilon or abs(x_next[1] - x_cur[1]) >= epsilon) and count <= max_turns: 
        x_last = x_cur
        x_cur = x_next
        x_next = None
        pk_last = pk_cur
        lt_last = lt_cur
        gk_last = gk_cur
        # construct beta_k
        lr_cur = cal_lr_straight_line(func=func, vars=vars, start_p=x_cur, A=A, debug=debug)
        if debug:
            print('lr_cur: {}'.format(lr_cur))
        delta_gk_last = A*(x_cur-x_last)
        subs_pairs = get_subs_pairs(vars, x_cur)
        gk_cur = _gk.subs(subs_pairs)
        if beta_mode=='H_S':
            numerator = delta_gk_last.T*gk_cur
            denominator = delta_gk_last.T*pk_last
            beta_k = numerator[0]/denominator[0]
        elif beta_mode=='F_R':
            # if debug:
                # print('gk_cur:{}, gk_cur.T:{}, gk_last: {}, gk_last.T: {}'.format(gk_cur, gk_cur.T, gk_last, gk_last.T))
            numerator = gk_cur.T*gk_cur
            denominator = gk_last.T*gk_last
            beta_k = numerator[0]/denominator[0]
        elif beta_mode=='P_R':
            numerator = delta_gk_last.T*gk_cur
            denominator = gk_last.T*gk_last
            beta_k = numerator[0]/denominator[0]
        print('beta_k:{}, numerator: {}, denominator: {}'.format(beta_k, numerator, denominator))
        pk_cur = -1*gk_cur + beta_k * pk_last
        print('pk_cur: {}'.format(pk_cur))
        x_next = x_cur+ lr_cur*pk_cur
        count += 1
        if debug:
            print('--------  {}  -------'.format(count))
            print('x_next: {}'.format(x_next))
    
    euclid_dis = sp.Pow(sp.Pow(x_next[0]-x_cur[0], 2) + sp.Pow(x_next[1]-x_cur[1], 2), 0.5)
    if euclid_dis <= epsilon:
        return {'convergence': True, 'point': x_next}
    else:
        return {'convergence': False, 'point': x_next}
    
def get_subs_pairs(vars, p):
    """ @vars, @the solid point """
    subs_pairs = []
    for i in range(len(p)):
        subs_pairs.append((vars[i], p[i]))
    return subs_pairs

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
    # x, y = sp.symbols('x, y')
    # vars = [x, y]
    # vec = sp.Matrix([[x], [y]])
    # A = sp.Matrix([[2, 1], [1, 2]])
    # f = 0.5*vec.T*A*vec
    # start_p = sp.Matrix([[0.8], [-0.25]])
    # lr = cal_lr_straight_line(f, [x, y], start_p=start_p)
    # print(lr)
    # subs_pairs = []
    # for i in range(len(vars)):
    #     subs_pairs.append((vars[i], start_p[i]))
    # _gk = multi_diff(f, [x, y])
    # gk = _gk.subs(subs_pairs)
    # next_p = start_p - gk*lr
    # print(next_p)

    ## newton method
    ## 1
    # x, y = sp.symbols('x, y')
    # f = x**2+25*y**2
    # start_p = sp.Matrix([[0.5], [0.5]])
    # end_p = newton_descent(f, [x, y], start_p)
    # print(end_p)
    ## 2
    # x, y = sp.symbols('x, y')
    # f = sp.Pow(y-x, 4) + 8*x*y - x + y + 3
    
    ## conjungate_gradient_descent
    x, y = sp.symbols('x, y')
    vars = [x, y]
    vec = sp.Matrix([[x], [y]])
    A = sp.Matrix([[2, 1], [1, 2]])
    f = 0.5*vec.T*A*vec
    start_p = sp.Matrix([[0.8], [-0.25]])
    ans = conjungate_gradient_descent(func=f, vars=vars, start_p=start_p, max_turns=3, beta_mode='F_R', A=A)
    print(ans)