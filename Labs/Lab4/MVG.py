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
    # print(log_C)
    log_2pi = math.log(2*math.pi)
    y = np.zeros(x.shape[1])
    for j in range(M):
        for i in range(x.shape[1]):
            # x_i = x[:, i]
            norm_x = x[j][i] - mu
            inter_value = np.dot(norm_x.T, inv_C)
            dot_mult = np.dot(inter_value, norm_x)
            MVG = (-M*log_2pi - log_C - dot_mult)/2
            # MVG = np.subtract((-1*M*np.log(2*np.pi))/2, np.subtract(log_C, dot_mult)/2)
            y[i] = MVG

    return y


if __name__ == '__main__':

    ### 1 DIMENSION CHECK ###

    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(ML.vrow(XPlot), m, C)))
    # plt.show()
    pdfSol = np.load('./llGAU.npy')
    pdfGau = logpdf_GAU_ND(ML.vrow(XPlot), m, C)
    print("1-D array absolute max error: ", end='')
    print(np.abs(pdfSol - pdfGau).max())

    XND = np.load('./XND.npy')
    print(XND.shape)
    mu = np.load('./muND.npy')
    C = np.load('./CND.npy')
    # print(C.shape)
    pdfSol = np.load('./llND.npy')
    # print(pdfSol.shape)
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(pdfGau.shape)
    print("2-D array absolute max error: ", end='')
    print(np.abs(pdfSol - pdfGau).max())


