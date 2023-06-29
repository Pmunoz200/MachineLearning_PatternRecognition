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


def polynomial_kernel(xi, xj, d, C, eps):
    interm = np.dot(xi.T, xj)
    interm += C
    G = np.add(np.power(interm, d), eps)
    return G


def radial_kernel(xi, xj, gamma, eps):
    G = np.ones((xi.shape[1], xj.shape[1]))
    for i in range(xi.shape[1]):
        for j in range(xj.shape[1]):
            absolute = np.linalg.norm(xi[:, i] - xj[:, j])
            G[i, j] = -gamma * np.square(absolute)
    G = np.add(np.exp(G), eps)
    return G


def dual_svm(
    alpha,
    training_att,
    training_labels,
    K=1,
    d=0,
    c=0,
    eps=0,
    gamma=0,
    funct="polynomial",
):
    kern = funct.lower()
    one = np.ones(training_att.shape[1])
    zi = 2 * training_labels - 1
    z = np.dot(zi, zi.T)
    if kern == "polynomial":
        G = polynomial_kernel(training_att, training_att, d, c, eps)
    elif kern == "radial":
        G = radial_kernel(training_att, training_att, gamma, eps=eps)
    else:
        D = np.ones(training_att.shape[1]) * K
        D = np.vstack((training_att, D))
        G = np.dot(D.T, D)
    H = np.zeros((training_att.shape[1], training_att.shape[1]))
    for i in range(training_att.shape[1]):
        for j in range(training_att.shape[1]):
            H[i, j] = zi[i] * zi[j]
            H[i, j] *= G[i, j]
    retFun = np.dot(alpha.T, H)
    retFun = np.dot(retFun, alpha) / 2
    retFun = retFun - np.dot(alpha.T, one)
    retGrad = np.dot(H, alpha)
    retGrad -= one
    return (retFun, retGrad)


def svm(
    training_att,
    training_labels,
    test_att,
    test_labels,
    constrain,
    dim=2,
    c=1,
    K=1,
    gamma=1,
    eps=0,
    model="",
):
    alp = np.ones(DTR.shape[1])
    constrain = np.array([(0, constrain)] * DTR.shape[1])
    [x, f, d] = scipy.optimize.fmin_l_bfgs_b(
        dual_svm,
        alp,
        args=(training_att, training_labels, K, dim, c, eps, gamma, model),
        bounds=constrain,
        factr=1,
    )
    zi = 2 * training_labels - 1
    print(f"Dual lost: {-f}")
    kern = model.lower()
    if kern == "polynomial":
        S = x * zi
        S = np.dot(S, polynomial_kernel(training_att, test_att, dim, c, eps))
    elif kern == "radial":
        S = x * zi
        S = np.dot(S, radial_kernel(training_att, test_att, gamma, eps))
    else:
        D = np.ones(training_att.shape[1]) * K
        D = np.vstack((training_att, D))
        w = x * zi
        w = w * D
        w = np.sum(w, axis=1)
        x_val = np.vstack((test_att, np.ones(test_att.shape[1]) * K))
        S = np.dot(w.T, x_val)
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
    accuracy = svm(
        DTR,
        LTR,
        DEV,
        LEV,
        constrain=1,
        dim=2,
        c=1,
        gamma=10,
        eps=1,
        K=1,
    )
    print((1 - accuracy) * 100)
