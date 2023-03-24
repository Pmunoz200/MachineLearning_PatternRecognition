import numpy as np
import general_methods
import scipy
from matplotlib import pyplot as plt


def covariance_within_class(matrix_values, label, class_labels):
    within_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    n = matrix_values.size
    for i in range(len(class_labels)):
        centered_matrix = general_methods.center_data(matrix_values[:, label == i])
        cov_matrix = general_methods.covariance(centered_matrix)
        cov_matrix = np.multiply(cov_matrix, centered_matrix.size)
        within_cov = np.add(within_cov, cov_matrix)
    within_cov = np.divide(within_cov, n)
    return within_cov


def covariance_between_class(matrix_values, label, class_labels):
    between_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    N = matrix_values.size
    m_general = general_methods.mean_of_matrix_rows(matrix_values)
    for i in range(len(class_labels)):
        values = matrix_values[:, label == i]
        nc = values.size
        m_class = general_methods.mean_of_matrix_rows(values)
        norm_means = np.subtract(m_class, m_general)
        matr = np.multiply(nc, np.dot(norm_means, norm_means.T))
        between_cov = np.add(between_cov, matr)
    between_cov = np.divide(between_cov, N)
    return between_cov


def between_within_covariance (matrix_values, label, class_labels):
    Sw = covariance_within_class(attributes, labels, class_labels)
    Sb = covariance_between_class(attributes, labels, class_labels)
    return Sw, Sb


def LDA_generalized_problem(matrix_values, label, class_labels, m):
    [Sw, Sb] = between_within_covariance(matrix_values, label, class_labels)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    print(W)
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    return W, U


if __name__ == '__main__':
    class_label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # Fill with class label names
    attribute_names = ["sepal length", "sepal width", "petal length", "petal width"]  # Fill with class attribute names
    pathname = '/Users/pablomunoz/Desktop/Polito 2023-1/MachineLearning/Codes/Labs/data/iris.csv'
    m = 2

    [attributes, labels] = general_methods.load(pathname, class_label, attribute_names)
    [Sw, Sb] = between_within_covariance(attributes, labels, class_label)
    [W, U] = LDA_generalized_problem(attributes, labels, class_label, m)
    P = np.dot(W.T, attributes)
    general_methods.graphic_scatter_2d(P, labels, class_label)




