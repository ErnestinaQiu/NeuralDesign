"""
author: Wenyu Qiu
des: some def for the diff of func
"""
import sympy as sp

def multi_diff(func, vars):
    """ acquire diff for n-dimension func """
    dif_vect = sp.zeros(len(vars), 1)
    for i in range(len(vars)):
        tmp_dif = sp.diff(func, vars[i])
        dif_vect[i] = tmp_dif
    return dif_vect

def hessian(func, vars):
    """ for n-dimension hessian matrix, @func(spy), @variables(under spy sys) """
    hes_mat = sp.zeros(len(vars), len(vars))
    for i in range(len(vars)):
        v1 = vars[i]
        df1 = sp.diff(func, v1)
        for j in range(len(vars)):
            v2 = vars[j]
            df2 = sp.diff(df1, v2)
            hes_mat[i, j] = df2
    return hes_mat

if __name__ == '__main__':
    x1, x2 = sp.symbols('x1, x2')
    func = x1**2 + x2**2
    # print(hessian(func, [x1, x2]))
    print(multi_diff(func, [x1, x2]))