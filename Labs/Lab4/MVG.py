import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MLandPattern import MLandPattern as ML
import time


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


if __name__ == '__main__':

    ### 1 DIMENSION CHECK ###

    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(ML.vrow(XPlot), m, C)))
    # plt.show()
    pdfSol = np.load('./llGAU.npy')
    start1 = time.time()
    pdfGau = logpdf_GAU_ND(ML.vrow(XPlot), m, C)
    end1 = time.time()
    start2 = time.time()
    pdfG = ML.logpdf_GAU_ND(ML.vrow(XPlot), m, C)
    end2 = time.time()
    print(f'Time optimized: {start1-end1}')
    print(f'Time normal: {start2 - end2}')
    print("1-D array absolute max error: ", end='')
    print(np.abs(pdfSol - pdfGau).max())

    XND = np.load('./XND.npy')
    mu = np.load('./muND.npy')
    C = np.load('./CND.npy')
    pdfSol = np.load('./llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(pdfGau.shape)
    print("2-D array absolute max error: ", end='')
    print(np.abs(pdfSol - pdfGau).max())


