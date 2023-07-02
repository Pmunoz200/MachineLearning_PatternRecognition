import json
import numpy as np
import scipy
import math
from MLandPattern import MLandPattern as ML
import sklearn.datasets


def calculate_model(S, test_points, model, prior_probability, test_labels=[]):
    model = model.lower()
    funct = lambda r: 1 if r > 0 else 0
    if model == "Generative":
        logSJoint = S + np.log(prior_probability)
        logSMarginal = ML.vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)
        predictions = np.argmax(SPost, axis=0)
    elif model == "gmm":
        logSMarginal = scipy.special.logsumexp(S, axis=0)
        logSPost = S - logSMarginal
        SPost = np.exp(logSPost)
        predictions = np.argmax(SPost, axis=0)
    elif model == "regression":
        predictions = np.array(list(map(funct, S)))
    else:
        predictions = np.array(list(map(funct, S)))
    if len(test_labels) != 0:
        error = np.abs(test_labels - predictions)
        error = np.sum(error)
    return predictions, (1 - error)


def load_iris_binary():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    # D = D[:, L != 0]  # We remove setosa from D
    # L = L[L != 0]  # We remove setosa from L
    # L[L == 2] = 0  # We assign label 0 to virginica (was label 2) return D, L
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


def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, "w") as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, "r") as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def logpdf_GMM(x, GMM):
    S = np.zeros((len(GMM), x.shape[1]))
    for i in range(len(GMM)):
        ll = ML.logpdf_GAU_ND(x, GMM[i][1], GMM[i][2])
        S[i, :] = ML.logpdf_GAU_ND(x, GMM[i][1], GMM[i][2])
        S[i, :] += np.log(GMM[i][0])
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def ll_gaussian(x, mu, C):
    M = mu[0].shape[0]
    s = []
    for i in range(len(C)):
        inv_C = np.linalg.inv(C[i])
        # print(inv_C.shape)
        [_, log_C] = np.linalg.slogdet(C[i])
        log_2pi = -M * math.log(2 * math.pi)
        x_norm = x - ML.vcol(mu[i])
        inter_value = np.dot(x_norm.T, inv_C)
        dot_mul = np.dot(inter_value, x_norm)
        dot_mul = np.diag(dot_mul)
        y = (log_2pi - log_C - dot_mul) / 2
        s.append(y)
    return s


def EM(x, mu, cov, w, threshold=10**-6, psi=0, diag=0, tied=0):
    delta = 100
    previous_ll = 10
    cont = 1000
    mu = np.array(mu)
    cov = np.array(cov)
    if diag:
        cov = cov * np.eye(cov.shape[1])
    if tied:
        cov[:] = np.sum(cov, axis=0) / x.shape[1]
    if psi:
        for i in range(cov.shape[0]):
            U, s, _ = np.linalg.svd(cov[i])
            s[s < psi] = psi
            cov[i] = np.dot(U, ML.vcol(s) * U.T)
    w = ML.vcol(np.array((w)))
    while (delta > threshold) and (cont > 0):
        #### E-STEP ####
        ll = np.array(ll_gaussian(x, mu, cov))
        SJoint = ll + np.log(w)
        logSMarginal = scipy.special.logsumexp(SJoint, axis=0)
        logSPost = SJoint - logSMarginal
        SPost = np.exp(logSPost)

        #### M - STEP ####
        fg = np.dot(SPost, x.T)
        zg = ML.vcol(np.sum(SPost, axis=1))
        sg = []
        n_mu = fg / zg
        new_C = []
        mul = []
        for i in range(mu.shape[0]):
            psg = np.zeros((x.shape[0], x.shape[0]))
            for j in range(x.shape[1]):
                xi = x[:, j].reshape((-1, 1))
                xii = np.dot(xi, xi.T)
                psg += SPost[i, j] * xii
            mul.append(np.dot(ML.vcol(n_mu[i, :]), ML.vcol(n_mu[i, :]).T))
            sg.append(psg)
        div = np.array(sg) / zg.reshape((-1, 1, 1))
        new_mu = np.array(n_mu)
        mul = np.array(mul)
        new_C = div - mul
        new_w = ML.vcol(zg / np.sum(zg, axis=0))
        if diag:
            new_C = new_C * np.eye(new_C.shape[1])
        if tied:
            new_C[:] = np.sum(new_C, axis=0) / x.shape[1]
        if psi:
            for i in range(new_C.shape[0]):
                U, s, _ = np.linalg.svd(new_C[i])
                s[s < psi] = psi
                new_C[i] = np.dot(U, ML.vcol(s) * U.T)
        previous_ll = np.sum(logSMarginal) / x.shape[1]
        s = np.array(ll_gaussian(x, new_mu, new_C))
        newJoint = s + np.log(new_w)
        new_marginal = scipy.special.logsumexp(newJoint, axis=0)
        avg_ll = np.sum(new_marginal) / x.shape[1]
        delta = abs(previous_ll - avg_ll)
        previous_ll = avg_ll
        mu = new_mu
        cov = new_C
        w = new_w
        cont -= 1
        # print(delta)
    return avg_ll, mu, cov, w


def GMM(
    train_data,
    train_labels,
    test_data,
    test_label,
    niter,
    alpha,
    threshold,
    psi=0,
    diag=0,
    tied=0,
):
    class_labels = np.unique(train_labels)
    cov = ML.multiclass_covariance(train_data, train_labels)
    multi_mu = ML.multiclass_mean(train_data, train_labels)
    densities = []
    class_mu = []
    class_c = []
    class_w = []
    for i in class_labels:
        [_, mu, cov, w] = LBG(
            train_data[:, train_labels == i],
            niter=niter,
            alpha=alpha,
            psi=psi,
            diag=diag,
            tied=tied,
        )
        class_mu.append(mu)
        class_c.append(cov)
        class_w.append(w)
    class_mu = np.array(class_mu)
    class_c = np.array(class_c)
    class_w = np.array(class_w)
    densities = []
    for i in class_labels:
        ll = np.array(ll_gaussian(test_data, class_mu[i], class_c[i]))
        Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
        logdens = scipy.special.logsumexp(Sjoin, axis=0)
        densities.append(logdens)
    S = np.array(densities)
    # SJoint = S + np.log(class_w)
    # logSMarginal = scipy.special.logsumexp(SJoint, axis=0)
    # logSPost = SJoint - logSMarginal
    # SPost = np.exp(logSPost)
    predictions = np.argmax(S, axis=0)
    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        # print(f'Accuracy: {acc}%')
        # print(f'Error: {(100 - acc)}%')

    return S, predictions, acc


def LBG(x, niter, alpha, psi=0, diag=0, tied=0):
    mu = ML.mean_of_matrix_rows(x)
    mu = mu.reshape((1, mu.shape[0], mu.shape[1]))
    C = ML.covariance(x)
    C = C.reshape((1, C.shape[0], C.shape[1]))
    w = np.ones(1).reshape(-1, 1, 1)
    if not niter:
        [ll, mu, C, w] = EM(x, mu, C, w, psi=psi, diag=diag, tied=tied)
        mu = mu.reshape((-1, mu.shape[1], 1))
        w = w.reshape((-1, 1, 1))
    # print(ll)
    new_gmm = []
    for i in range(niter):
        new_mu = []
        new_cov = []
        new_w = []
        for i in range(len(mu)):
            U, s, _ = np.linalg.svd(C[i])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_w.append(w[i] / 2)
            new_w.append(w[i] / 2)
            new_mu.append(mu[i] + d)
            new_mu.append(mu[i] - d)
            new_cov.append(C[i])
            new_cov.append(C[i])
        [ll, mu, C, w] = EM(x, new_mu, new_cov, new_w, psi=psi, diag=diag, tied=tied)
        mu = mu.reshape((-1, mu.shape[1], 1))
        w = w.reshape((-1, 1, 1))
    # print(gmm)
    return ll, mu, C, w


if __name__ == "__main__":
    [D, L] = load_iris_binary()
    (DTR, LTR), (DEV, LEV) = ML.split_db(D, L, 2 / 3)
    print("Full")
    for i in range(4):
        [_, _, acc] = GMM(DTR, LTR, DEV, LEV, i, 0.1, 10**-6, psi=0.01)
        print(100 - acc)
    print("Diagonal")
    for i in range(4):
        [_, _, acc] = GMM(DTR, LTR, DEV, LEV, i, 0.1, 10**-6, diag=1, psi=0.01)
        print(100 - acc)
    print("Tied")
    for i in range(4):
        [_, _, acc] = GMM(DTR, LTR, DEV, LEV, i, 0.1, 10**-6, tied=1, psi=0.01)
        print(100 - acc)

    # path = "data/GMM_data_4D.npy"
    # X = np.load(path)
    # gmm = load_gmm("data/GMM_4D_3G_init.json")
    # densities = np.load("data/GMM_4D_3G_init_ll.npy")
    # print(densities.shape)
    # mu = []
    # cov = []
    # w = []
    # for i in range(len(gmm)):
    #     w.append(gmm[i][0])
    #     mu.append(gmm[i][1])
    #     cov.append(gmm[i][2])
    # mu = np.array(mu)
    # cov = np.array(cov)
    # w = ML.vcol(np.array((w)))
    # ll = np.array(ll_gaussian(X, mu, cov))
    # SJoint = ll + np.log(w)
    # logSMarginal = scipy.special.logsumexp(SJoint)
    # # print(y.shape)
    # print(np.max(SJoint - densities))

    # [ll, _, _, _] = EM(X, mu, cov, w)
    # print(ll)
    # teor_GMM_3G = load_gmm("data/GMM_4D_3G_EM.json")
    # [ll, mu, cov, w] = LBG(X, 2, 0.1, psi=0.01)
    # print(ll, end="\n\n\n")
    # path = "data/GMM_data_1D.npy"
    # X = np.load(path)
    # [ll, mu, cov, w] = LBG(X, 2, 0.1)
    # print(ll)
    # LBG_gmm = load_gmm("data/GMM_1D_4G_EM_LBG.json")
    # for i in range(len(LBG_gmm)):
    #     for j in range(3):
    #         print(gmm_LBG[i][j] - LBG_gmm[i][j])
    #         print()
