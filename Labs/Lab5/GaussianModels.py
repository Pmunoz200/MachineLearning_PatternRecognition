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


def multiclass_covariance(matrix, labels):
    class_labels = np.unique(labels)
    # print(class_labels.size)
    within_cov = np.zeros((class_labels.size, matrix.shape[0], matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        centered_matrix = ML.center_data(matrix[:, labels == i])
        within_cov[i, :, :] = ML.covariance(centered_matrix)
    return within_cov


def multiclass_mean(matrix, labels):
    class_labels = np.unique(labels)
    # print(class_labels.size)
    multi_mu = np.zeros((class_labels.size, matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        mu = ML.mean_of_matrix_rows(matrix[:, labels == i])
        multi_mu[i, :] = mu[:, 0]
    return multi_mu


def logLikelihood(X, mu, c, tot=0):
    """
    Calculates the Logarithmic Maximum Likelihood estimator
    :param X: matrix of the datapoints of a dataset, with a size (n x m)
    :param mu: row vector with the mean associated to each dimension
    :param c: Covariance matrix
    :param tot: flag to define if it returns value per datapoint, or total sum of logLikelihood
    :return: the logarithm of the likelihood of the datapoints, and the associated gaussian density
    """
    M = c.shape[1]
    logN = ML.logpdf_GAU_ND(X, mu, c)
    if tot:
        return logN.sum()
    else:
        return logN


def MVG_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(
            np.exp(logLikelihood(test_data, ML.vcol(multi_mu[i, :]), cov[i]))
        )
    S = np.array(densities)
    SJoint = S * prior_probability
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

    return S, SPost, predictions, acc


def MVG_log_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, ML.vcol(multi_mu[i, :]), cov[i]))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = ML.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
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
    # SJoint_MVG = np.load('Solutions/SJoint_MVG.npy')
    # # print(DTR.shape, LTR.shape)
    # [SJoint, SPost] = MVG_classifier(DTR, LTR, DTE, prior_prob)
    # print("SJoint error: ", end='')
    # print(np.abs(SJoint - SJoint_MVG).max())
    [SPost, Predictions] = MVG_classifier(DTR, LTR, DTE, prior_prob, LTE)
    [SPost, predictions] = MVG_log_classifier(DTR, LTR, DTE, prior_prob, LTE)

    # logSJ_MVG = np.load('Solutions/logSJoint_MVG.npy')
    # logSP_MVG = np.load('Solutions/logPosterior_MVG.npy')
    # logSM_MVG = np.load('Solutions/logMarginal_MVG.npy')

    # print(np.abs(logSJoint - logSJ_MVG).max())
    # print(np.abs(logSMarginal - logSM_MVG).max())
    # print(np.abs(logSPost - logSP_MVG).max())
