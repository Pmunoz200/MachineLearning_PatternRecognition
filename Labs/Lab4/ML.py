import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MLandPattern import MLandPattern as ML


def logpdf_GAU_ND(x, mu, C):
    M = C.shape[1]
    inv_C = np.linalg.inv(C)
    # print(inv_C.shape)
    [_, log_C] = np.linalg.slogdet(C)

    #print(log_C)
    log_2pi = -M * math.log(2*math.pi)
    x_norm = x-mu
    inter_value = np.dot(x_norm.T, inv_C)
    dot_mul = np.dot(inter_value, x_norm)
    dot_mul = np.diag(dot_mul)

    y = (log_2pi - log_C - dot_mul)/2
    return y


def logLikelihood (X, mu, c):
    M = c.shape[1]
    logN = logpdf_GAU_ND(X, mu, c)
    print(logN.shape)

    acum = logN.sum()
    return acum




if __name__ == '__main__':
    XND = np.load('./XND.npy')
    # print(XND.shape)
    m_ML = ML.mean_of_matrix_rows(XND)
    print(m_ML)
    C_ML = ML.covariance(ML.center_data(XND))
    print(C_ML)
    # print(m_ML)
    # print(cov)
    ll = logLikelihood(XND, m_ML, C_ML)
    print(ll)

    X1D = np.load('X1D.npy')
    m_ML1 = ML.mean_of_matrix_rows(X1D)
    print(m_ML1)
    C_ML1 = ML.covariance(ML.center_data(X1D))
    print(C_ML1)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(ML.vrow(XPlot), m_ML1, C_ML1)))
    # plt.show()

    ll = logLikelihood(X1D, m_ML1, C_ML1)
    print(ll)


