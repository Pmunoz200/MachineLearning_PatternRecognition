import sys
import numpy as np
import MLandPattern.MLandPattern as ML


def load_data():
    lInf = []

    f = open("data/inferno.txt", encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open("data/purgatorio.txt", encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open("data/paradiso.txt", encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest


def extractData(train_data, epsilon):
    mappingWords = []
    occurences = []
    for line in train_data:
        for word in line.split(" "):
            if word not in mappingWords:
                mappingWords.append(word)
                occurences.append(1)
                continue
            occurences[mappingWords.index(word)] += 1 + epsilon

    return occurences, mappingWords


if __name__ == "__main__":
    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    # print(lInf_train)

    occInf, mappingInf = extractData(lInf_train, 0.001)

    print(sum(occInf))
