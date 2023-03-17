import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
class_label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
attribute_names = ["sepal length", "sepal width", "petal length", "petal width"]
alpha_val = 0.5


def load(pathname):
    df = pd.read_csv(pathname, header=None)
    print(df.head())
    attribute = np.array(df.iloc[:, 0:4])
    attribute = attribute.T
    # print(attribute)
    label_list = []
    # print(label.size)
    for lab in df.loc[:, 4]:
        label_list.append(class_label.index(lab))
    label = np.array(label_list)
    return attribute, label


def histogram_1n(setosa, versicolor, virginica, x_axis='', y_axis=''):
    plt.hist(setosa, color='blue', alpha=alpha_val, label=class_label[0], density=True)
    plt.hist(versicolor, color='orange', alpha=alpha_val, label=class_label[1], density=True)
    plt.hist(virginica, color='green', alpha=alpha_val, label=class_label[2], density=True)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def scatter_2d(setosa, versicolor, virginica, x_axis = '', y_axis = ''):
    plt.scatter(setosa[0], setosa[1], c='blue', s=1.5)
    plt.scatter(versicolor[0], versicolor[1], c='orange', s=1.5)
    plt.scatter(virginica[0], virginica[1], c='green', s=1.5)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

[attributes, labels] = load("iris.csv")
print(f"Attributes size: {attributes.size}")
print(f"Attributes shape: {attributes.shape}")
print(f"Labels size: {labels.size}")
print(f"Labels shape: {labels.shape}")


def graficar():
    values_histogram = {}

    for i in range(len(attribute_names)):
        values_histogram[attribute_names[i]] = [attributes[i, labels == 0], attributes[i, labels == 1], attributes[i, labels == 2]]

    for a in attribute_names:
        histogram_1n(values_histogram[a][0], values_histogram[a][1], values_histogram[a][2], x_axis=a)

    for xk, xv in values_histogram.items():
        cont = 0
        for yk, yv in values_histogram.items():
            if xk == yk:
                plt.subplot(141+cont)
                histogram_1n(xv[0], xv[1], xv[2], x_axis=xk)
                cont += 1
            else:
                plt.subplot(141+cont)
                scatter_2d([xv[0], yv[0]], [xv[1], yv[1]], [xv[2], yv[2]], x_axis=xk, y_axis=yk)
                cont += 1
        plt.show()


def mcol(matrix, vector):
    column_vector = vector.reshape((matrix.shape[0], 1))
    return column_vector


def mrow(matrix, vector):
    row_vector = vector.reshape((1, matrix.shape[0]))
    return  row_vector


mu = attributes.mean(1)
muCol = mcol(attributes, mu)
normalized_attributes = attributes - muCol
print(mu.shape)
print(normalized_attributes)

