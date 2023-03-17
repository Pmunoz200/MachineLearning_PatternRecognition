import numpy as np
import pandas as pd

class_label = []  # Fill with class label names
attribute_names = []  # Fill with class attribute names


#  Loads a dataset and divides them in Attributes and Classes
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


#  Reshapes a vector into a col
def vcol(vector):
    column_vector = vector.reshape((vector.size, 1))
    return column_vector


def vrow(vector):
    row_vector = vector.reshape((1, vector.size))
    return row_vector

