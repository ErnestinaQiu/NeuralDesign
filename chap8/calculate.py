"""
date: 20211017
author: Wenyu Qiu
des: 
"""
import numpy as np
import sympy as sp
from sympy.abc import pi

def hessian(expr, vars):
    """ figure out the hessian matrix of an equation 
    @expr(expr of sympy), @vars(a list of the symbols one)
    """
    cols_num = len(vars)
    hes_mat = sp.zeros(cols_num)
    for i in range(cols_num):
        tmp_dif = sp.diff(expr, vars[i])    
        # print('dif {}: {}'.format(i, tmp_dif))
        for j in range(cols_num):
            # print('dif {}{}: {}'.format(i, j, tmp_dif))
            hes_mat[i, j] = sp.diff(tmp_dif, vars[j])
    return hes_mat

def _8_4(x0):
    #p8.4, F(x) = cos(x)
    x = sp.symbols('x')
    expr = sp.cos(x)
    taylor_sery = expr.series(x, x0, 3)
    print(taylor_sery)
    return taylor_sery

def _8_5():
    def verify_minimum(hesmat, point):
        ans = 1  # if ans == 3, there is at least one eigenvalue below 0 
        hesmat = hesmat.subs([(x1,point[0]), (x2,point[1])])
        eigen_results = hesmat.eigenvects()
        ans = 1  # if ans == 3, there is at least one eigenvalue below 0 
        for i in range(len(eigen_results)):
            eigen_result = eigen_results[i]
            eigenvalue = eigen_result[0]
            if eigenvalue < 0:
                ans = 3
                break
        return ans

    x1, x2 = sp.symbols('x1, x2')
    expr = (x2 - x1)**4 + 8*x1*x2 - x1 + x2 + 3
    hesmat = hessian(expr, [x1, x2])
    local_minimum = []
    points = [[-0.41878, 0.41878], [-0.134797, 0.134797], [0.55358, -0.55358]]
    
    for p in points:
        ans = verify_minimum(hesmat=hesmat, point=p)
        if ans == 1:
            local_minimum.append(p)
        
    return local_minimum

if __name__ == '__main__':
    # _8_4(2)
    
    # 8_5
    print(_8_5()) 
    
