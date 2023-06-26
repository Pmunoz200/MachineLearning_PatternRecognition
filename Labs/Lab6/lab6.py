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


def extractData(train_data):
    size = train_data.shape[0]
    mappingWords = []
    for i in range(size):
        for line in train_data[i]:
            for word in line.split(" "):
                if word not in mappingWords:
                    mappingWords.append(word)
                    continue
    occurences = np.zeros((size, len(mappingWords)))

    for i in range(size):
        for line in train_data[i]:
            for word in line.split(" "):
                occurences[i, mappingWords.index(word)] += 1

    return occurences, mappingWords


def ML_model(occurencesAttr, mappingWords, validation_set, prior_prob, epsilon=0.001):
    total_words = ML.vcol(np.sum(occurencesAttr, axis=1))
    size = occurencesAttr.shape[0]
    model = np.divide(occurencesAttr, total_words) + epsilon
    densities = ML.vcol(np.zeros(size))
    for line in validation_set:
        for word in line:
            if word in mappingWords:
                # print(model[:, mappingWords.index(word)])
                densities = densities + np.log(
                    ML.vcol(model[:, mappingWords.index(word)])
                )
                densities *= ML.vcol(model[:, mappingWords.index(word)])
    densities += np.log(prior_prob)

    return model, np.exp(densities)


if __name__ == "__main__":
    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    attributes = np.array([lInf_train, lPur_train, lPar_train])
    evalutation = [lInf_evaluation, lPur_evaluation, lPar_evaluation]
    priorprob = ML.vcol(np.array([1 / 3, 1 / 3, 1 / 3]))
    # print(size(attributes))

    occurences, mapping = extractData(attributes)
    # print(occurences)

    model, densities = ML_model(
        occurences, mapping, lInf_evaluation, prior_prob=priorprob
    )

    print(densities)
