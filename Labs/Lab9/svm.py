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


def dual_svm(alpha, training_att, training_labels, K):
    D = np.ones(training_att.shape[1]) * K
    D = np.vstack((training_att, D))
    one = np.ones(training_att.shape[1])
    zi = 2 * training_labels - 1
    z = np.dot(zi, zi.T)
    G = np.dot(D.T, D)
    H = np.zeros((D.shape[1], D.shape[1]))
    for i in range(D.shape[1]):
        for j in range(D.shape[1]):
            H[i, j] = zi[i] * zi[j]
            H[i, j] *= G[i, j]
    retFun = np.dot(alpha.T, H)
    retFun = np.dot(retFun, alpha) / 2
    retFun = retFun - np.dot(alpha.T, one)
    retGrad = np.dot(H, alpha)
    retGrad -= one
    return (retFun, retGrad)


def primal_svm(w, training_att, training_labels, C):
    n_w = np.linalg.norm(w)
    n_w = np.power(n_w, 2)
    n_w /= 2
    zi = 2 * training_labels - 1
    inter_sol = np.dot(w.T, training_att)
    inter_sol = zi * inter_sol
    comp = np.ones(inter_sol.size) - inter_sol
    comp[comp < 0] = 0
    J = C * np.sum(comp)
    J += n_w
    return J


def svm(training_att, training_labels, K, test_att, test_labels, C):
    alp = np.ones(DTR.shape[1])
    D = np.ones(training_att.shape[1]) * K
    D = np.vstack((training_att, D))
    constrain = np.array([(0, C)] * DTR.shape[1])
    [x, f, d] = scipy.optimize.fmin_l_bfgs_b(
        dual_svm,
        alp,
        args=(training_att, training_labels, K),
        bounds=constrain,
        factr=1,
    )
    zi = 2 * training_labels - 1
    w = x * zi
    w = w * D
    w = np.sum(w, axis=1)
    # w, b = w[0:-1], w[-1]
    x_val = np.vstack((test_att, np.ones(test_att.shape[1]) * K))
    S = np.dot(w.T, x_val)
    primal_f = primal_svm(w, D, training_labels, C)
    print(f"Dual lost: {-f}")
    print(f"Primal lost: {primal_f}")
    print(primal_f + f)
    funct = lambda s: 1 if s > 0 else 0
    predictions = np.array(list(map(funct, S)))
    acc = 0
    for i in range(test_labels.shape[0]):
        if predictions[i] == test_labels[i]:
            acc += 1
    acc /= test_labels.size

    return acc


if __name__ == "__main__":
    [D, L] = load_iris_binary()
    (DTR, LTR), (DEV, LEV) = ML.split_db(D, L, 2 / 3)
    alp = np.ones(DTR.shape[1])
    accuracy = svm(DTR, LTR, 1, DEV, LEV, 10)
    print((1 - accuracy) * 100)
