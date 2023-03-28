import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

# class_label = []  # Fill with class label names
# attribute_names = []  # Fill with class attribute names


#  Loads a dataset and divides them in Attributes and Classes
def load(pathname, class_label, attribute_names):
    df = pd.read_csv(pathname, header=None)
    print(df.head())
    print("-"*50)
    attribute = np.array(df.iloc[:, 0:len(attribute_names)])
    attribute = attribute.T
    # print(attribute)
    label_list = []
    # print(label.size)
    for lab in df.loc[:, len(attribute_names)]:
        label_list.append(class_label.index(lab))
    label = np.array(label_list)
    return attribute, label


#  Reshapes a vector into a column vector
def vcol(vector):
    column_vector = vector.reshape((vector.size, 1))
    return column_vector


#  Reshapes a column vector into a row vector
def vrow(vector):
    row_vector = vector.reshape((1, vector.size))
    return row_vector


#  Calculates the mean of the rows of a matrix, and returns a column vector
def mean_of_matrix_rows(m):
    mu = m.mean(1)
    mu_col = vcol(mu)
    return mu_col


#  Normalizes the data by subtracting to each attribute its mean
def center_data(m):
    mean = mean_of_matrix_rows(m)
    centered_data = m - mean
    return centered_data


#  Calculates the Sample Covariance Matrix
def covariance(centered_matrix):
    n = centered_matrix.shape[1]
    cov = np.dot(centered_matrix, centered_matrix.T)
    cov = np.multiply(cov, 1/n)
    return cov


#  Calculates the eigen value and vectors for a matrix and returns them in ascending order
def eigen(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        s, U = np.linalg.eigh(matrix)
        return s, U
    else:
        s, U = np.linalg.eig(matrix)
        return s, U


#  Calculates the PCA dimension reduction to an m-dimension space, returning the projection matrix P
def PCA(attribute_matrix, m):
    mu = mean_of_matrix_rows(attribute_matrix)
    DC = center_data(attribute_matrix)
    C = covariance(DC)
    s, U = eigen(C)
    P = U[:, ::-1][:, 0:m]
    return P


def covariance_within_class(matrix_values, label, class_labels):
    within_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    n = matrix_values.size
    for i in range(len(class_labels)):
        centered_matrix = center_data(matrix_values[:, label == i])
        cov_matrix = covariance(centered_matrix)
        cov_matrix = np.multiply(cov_matrix, centered_matrix.size)
        within_cov = np.add(within_cov, cov_matrix)
    within_cov = np.divide(within_cov, n)
    return within_cov


def covariance_between_class(matrix_values, label, class_labels):
    between_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    N = matrix_values.size
    m_general = mean_of_matrix_rows(matrix_values)
    for i in range(len(class_labels)):
        values = matrix_values[:, label == i]
        nc = values.size
        m_class = mean_of_matrix_rows(values)
        norm_means = np.subtract(m_class, m_general)
        matr = np.multiply(nc, np.dot(norm_means, norm_means.T))
        between_cov = np.add(between_cov, matr)
    between_cov = np.divide(between_cov, N)
    return between_cov


def between_within_covariance (matrix_values, label, class_labels):
    Sw = covariance_within_class(matrix_values, label, class_labels)
    Sb = covariance_between_class(matrix_values, label, class_labels)
    return Sw, Sb


def LDA1(matrix_values, label, class_labels, m):
    [Sw, Sb] = between_within_covariance(matrix_values, label, class_labels)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    print(W)
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    return W, U


#  General method to graph a class-related data into a 2d scatter plot
def graphic_scatter_2d(matrix, labels, names, x_axis="Axis 1", y_axis="Axis 2"):
    for i in range(len(names)):
        plt.scatter(matrix[0][labels == i], matrix[1][labels == i], label=names[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()


