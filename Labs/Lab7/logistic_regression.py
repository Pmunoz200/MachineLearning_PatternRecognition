import numpy as np
import scipy.optimize


def fun(values):
    retFun = np.power(values[0] + 3, 2) + np.sin(values[0]) + np.power(values[1] + 1, 2)
    return retFun


def fun_and_grradient(values):
    retFun = np.power(values[0] + 3, 2) + np.sin(values[0]) + np.power(values[1] + 1, 2)
    retGradient = 2 * (values[1] + 1)
    return np.array([retFun, retGradient])


def logistic_regretion(funct):
    x, f, d = scipy.optimize.fmin_l_bfgs_b(funct, np.array([0, 0]), approx_grad=True)
    return x, f, d


def logistic_regretion2(funct):
    x, f, d = scipy.optimize.fmin_l_bfgs_b(funct, np.array([0, 0]))
    return x, f, d


position, min_value, extra = logistic_regretion(fun)
print(position)


pos2, min_value2, _ = logistic_regretion2(fun_and_grradient)
print(pos2)
