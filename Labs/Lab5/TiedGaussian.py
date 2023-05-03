import math
import sklearn.datasets
import numpy as np
import MLandPattern.MLandPattern as ML
import scipy


def load_iris():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def TiedGaussian(train_data, train_labels, test_data, prior_probability, test_label=[]):
    class_labels = np.unique(train_labels)
    with_cov = ML.covariance_within_class(train_data, train_labels)
    print(with_cov.shape)
    multi_mu = ML.multiclass_mean(train_data, train_labels)
    # print(multi_mu[0])
    densities = []
    for i in range(class_labels.size):
        densities.append(
            np.exp(ML.logLikelihood(test_data, ML.vcol(multi_mu[i, :]), with_cov))
        )
    S = np.array(densities)

    SJoint = S * prior_probability
    SJoint_tied = np.load("Solutions/SJoint_TiedMVG.npy")
    print(np.abs(SJoint_tied - SJoint).max())
    print(SJoint.shape)
    SMarginal = ML.vrow(SJoint.sum(0))
    print(SMarginal.shape)
    SPost = SJoint / SMarginal
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(LTE)):
            if predictions[i] == LTE[i]:
                acc += 1
        acc /= len(LTE)
        print(f"Accuracy: {acc}")
        print(f"Error: {1 - acc}")

    return SPost, predictions


if __name__ == "__main__":
    [D, L] = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    prior_prob = ML.vcol(np.ones(3) * (1 / 3))

    [SPost, predictions] = TiedGaussian(DTR, LTR, DTE, prior_prob, LTE)
