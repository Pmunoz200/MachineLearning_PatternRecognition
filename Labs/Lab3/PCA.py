import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class_label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # Fill with class label names
attribute_names = ["sepal length", "sepal width", "petal length", "petal width"]  # Fill with class attribute names


def load(pathname):
    df = pd.read_csv(pathname, header=None)
    print(df.head())
    attribute = np.array(df.iloc[:, 0:len(attribute_names)])
    attribute = attribute.T
    # print(attribute)
    label_list = []
    # print(label.size)
    for lab in df.loc[:, len(attribute_names)]:
        label_list.append(class_label.index(lab))
    label = np.array(label_list)
    return attribute, label


def vcol(vector):
    column_vector = vector.reshape((vector.size, 1))
    return column_vector


def mean_of_matrix(m):
    mu = m.mean(1)
    mu_col = vcol(mu)
    return mu_col


def center_data(m):
    mean = mean_of_matrix(m)
    centered_data = m - mean
    return centered_data


def covariance(centered_matrix):
    n = centered_matrix.shape[1]
    cov = np.dot(centered_matrix, centered_matrix.T)
    cov = np.multiply(cov, 1/n)
    return cov


def eigen(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        s, U = np.linalg.eigh(matrix)
        return s, U


def PCA(attribute_matrix, m):
    mu = mean_of_matrix(attribute_matrix)
    DC = center_data(attribute_matrix)
    C = covariance(DC)
    s, U = eigen(C)
    P = U[:, ::-1][:, 0:m]
    return P


if __name__ == '__main__':
    pathname = '/Users/pablomunoz/Desktop/Polito 2023-1/MachineLearning/Codes/Labs/data/iris.csv'
    [attributes, lables] = load(pathname)
    m = 2
    P = PCA(attributes, m)
    x = np.dot(P.T, attributes)

    plt.scatter(x[0][lables == 0], x[1][lables == 0], c='orange')
    plt.scatter(x[0][lables == 1], x[1][lables == 1], c='blue')
    plt.scatter(x[0][lables == 2], x[1][lables == 2], c='green')
    plt.show()


