import numpy as np
import scipy.optimize
import MLandPattern.MLandPattern as ML
import sklearn.datasets


def load_iris_binary():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2) return D, L
    return D, L


def logreg_obj(v, DTR, LTR, l):
    n = DTR.shape[1]
    w, b = v[0:-1], v[-1]
    log_sum = 0
    for i in range(n):
        zi = 1 if LTR[i] else -1
        inter_sol = -zi * (np.dot(w.T, DTR[:, i]) + b)
        log_sum += np.logaddexp(0, inter_sol)
    retFunc = l / 2 * np.power(np.linalg.norm(w), 2) + 1 / n * log_sum
    return retFunc


def binaryRegression(train_attributes, train_labels, l, test_attributes, test_labels):
    x0 = np.zeros(train_attributes.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj, x0, approx_grad=True, args=(train_attributes, train_labels, l)
    )
    w, b = x[0:-1], x[-1]
    S = np.dot(w.T, test_attributes) + b
    funct = lambda s: 1 if s > 0 else 0
    predictions = np.array(list(map(funct, S)))
    acc = 0
    for i in range(test_labels.shape[0]):
        if predictions[i] == test_labels[i]:
            acc += 1
    acc /= test_labels.size
    return w, S, acc


if __name__ == "__main__":
    [D, L] = load_iris_binary()
    (DTR, LTR), (DEV, LEV) = ML.split_db(D, L, 2 / 3)
    [w, S, acc] = binaryRegression(DTR, LTR, 0.000001, DEV, LEV)
    print(1 - acc)
    [w, S, acc] = binaryRegression(DTR, LTR, 0.001, DEV, LEV)
    print(1 - acc)
    [w, S, acc] = binaryRegression(DTR, LTR, 0.1, DEV, LEV)
    print(1 - acc)
    [w, S, acc] = binaryRegression(DTR, LTR, 1, DEV, LEV)
    print(1 - acc)
