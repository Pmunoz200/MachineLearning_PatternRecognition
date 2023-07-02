import numpy
import json
import scipy
import os
import sys

sys.path.append(os.path.abspath("snakeMLpj"))
from snakeMLpj import density_estimation, numpy_transformations
import math


def load_gmm(filename):
    with open(filename, "r") as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]


def ll_gaussian(xtest, mu, C, log=True):
    score = []
    mu = numpy.array(mu)
    for i in range(len(C)):
        C_inv = numpy.linalg.inv(C[i])
        det = numpy.linalg.slogdet(C[i])[1]
        M = xtest.shape[0]
        log_pi = math.log(2 * math.pi)
        x_mu = numpy.subtract(xtest, mu[i].reshape((-1, 1)))
        r1 = numpy.dot(x_mu.T, C_inv)
        r2 = numpy.diagonal(numpy.dot(r1, x_mu))
        result = (-M * log_pi - det - r2) / 2
        if log:
            score.append(result)
        else:
            score.append(numpy.exp(result))
    return score


def EM(
    D, mus, covs, weights, threshold=10 ** (-6), psi=None, diagonal=False, tied=False
):
    l_old = -numpy.inf
    lg = numpy.inf
    g = len(mus)
    mu_1 = numpy.array(mus)
    cov_1 = numpy.array(covs)
    w_1 = numpy.array(weights)
    ll = ll_gaussian(D, mu_1, cov_1)
    Sj = density_estimation.SJoint(ll, pi=w_1)
    Marginal = density_estimation.SMarginal(Sj)
    r = density_estimation.SPost(Sj, Marginal, exp=True)
    Zg = r.sum(axis=1).reshape((-1, 1))
    if diagonal:
        covnew = []
        for k in range(g):
            Sigma_g = cov_1[k, :, :] * numpy.eye(cov_1[k, :, :].shape[0])
            covnew.append(Sigma_g)
        cov_1 = covnew
    if tied:
        covnew = []
        for k in range(g):
            Sigma_g = Zg[g, :, :] * cov_1[k, :, :]
            covnew.append(Sigma_g)
        cov_1 = covnew / D.shape[1]
    if psi:
        covnew = []
        for k in range(g):
            U, s, _ = numpy.linalg.svd(cov_1[k, :, :])
            s[s < psi] = psi
            covnew.append(numpy.dot(U, numpy_transformations.mcol(s) * U.T))
        cov_1 = covnew
    while lg >= threshold:
        ll = ll_gaussian(D, mu_1, cov_1)
        Sj = density_estimation.SJoint(ll, pi=w_1)
        Marginal = density_estimation.SMarginal(Sj)
        r = density_estimation.SPost(Sj, Marginal, exp=True)
        Fg = numpy.dot(r, D.T)
        Zg = r.sum(axis=1).reshape((-1, 1))
        mu_1 = Fg / Zg
        w_1 = Zg
        w_1 = (w_1 / w_1.sum()).reshape((-1, 1))
        Sg = []
        cov_1 = []
        b = []
        for i in range(g):
            psg = numpy.zeros((D.shape[0], D.shape[0]))
            for j in range(D.shape[1]):
                y = r[i, j]
                xi = D[:, j].reshape((-1, 1))
                xii = numpy.dot(xi, xi.T)
                psg += y * xii
            Sg.append(psg)
            b.append(
                numpy.dot(mu_1[i, :].reshape((-1, 1)), mu_1[i, :].reshape((1, -1)))
            )
        Sg = numpy.array(Sg)
        a = Sg / Zg.reshape((-1, 1, 1))
        b = numpy.array(b)
        cov_1 = a - b
        if diagonal:
            covnew = []
            for k in range(g):
                Sigma_g = cov_1[k, :, :] * numpy.eye(cov_1[k, :, :].shape[0])
                covnew.append(Sigma_g)
            cov_1 = covnew
        if tied:
            covnew = []
            for k in range(g):
                Sigma_g = Zg[g, :, :] * cov_1[k, :, :]
                covnew.append(Sigma_g)
            cov_1 = covnew / D.shape[1]
        if psi:
            covnew = []
            for k in range(g):
                U, s, _ = numpy.linalg.svd(cov_1[k, :, :])
                s[s < psi] = psi
                covnew.append(numpy.dot(U, numpy_transformations.mcol(s) * U.T))
            cov_1 = covnew
        l = Marginal.mean()
        lg = l - l_old
        l_old = l
        print("loss:", lg)
    return mu_1, cov_1, w_1


def LBG(D, alpha=0.1, num_splits=2, psi=None):
    mu, C = numpy_transformations.mean_cov(D)
    mu = numpy.array([mu])
    C = numpy.array([C])
    w = numpy.array([1]).reshape((-1, 1, 1))
    # print(mu.shape, C.shape, w.shape)
    for j in range(num_splits):
        mu_split = []
        C_split = []
        w_split = []
        for i in range(len(mu)):
            U, s, Vh = numpy.linalg.svd(C[i])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            mu_split.append(mu[i, :, :] - d)
            mu_split.append(mu[i, :, :] + d)
            C_split.append(C[i, :, :])
            C_split.append(C[i, :, :])
            w_split.append(w[i, :, :] / 2)
            w_split.append(w[i, :, :] / 2)
        mu, C, w = EM(D, mu_split, C_split, w_split, psi=psi)
        mu = numpy.array(mu).reshape((-1, D.shape[0], 1))
        C = numpy.array(C)
        w = numpy.array(w).reshape((-1, 1, 1))
        # print(mu.shape, C.shape, w.shape)
    return mu, C, w


def EM_long(D, mus, covs, weights, threshold=10 ** (-6)):
    l_old = 0
    lg = numpy.inf
    g = len(mus)
    mu_1 = mus
    cov_1 = covs
    w_1 = weights
    while lg >= threshold:
        ll = ll_gaussian(D, mu_1, cov_1)
        Sj = density_estimation.SJoint(ll, pi=w_1)
        Marginal = density_estimation.SMarginal(Sj)
        r = density_estimation.SPost(Sj, Marginal, exp=True)
        Sg = []
        Zg = []
        Fg = []
        mu_1 = []
        cov_1 = []
        w_1 = []
        for i in range(g):
            psg = numpy.zeros((D.shape[0], D.shape[0]))
            pzg = 0
            pfg = 0
            for j in range(D.shape[1]):
                y = r[i, j]
                pzg += y
                xi = D[:, j].reshape((-1, 1))
                pfg += y * xi
                xii = numpy.dot(xi, xi.T)
                psg += y * xii
            Zg.append(pzg)
            Fg.append(pfg)
            Sg.append(psg)
            pmg = pfg / pzg
            a = psg / pzg
            b = numpy.dot(pmg, pmg.T)
            pcg = a - b
            mu_1.append(pmg)
            cov_1.append(pcg)
            w_1.append(pzg)
        w_1 = numpy.array(w_1)
        w_1 = (w_1 / w_1.sum()).reshape((-1, 1))
        l = Marginal.mean()
        lg = l - l_old
        l_old = l
        print("loss:", lg)
    return mu_1, cov_1, w_1


def avg_ll(mus, covs, weights):
    ll = ll_gaussian(D, mus, covs)
    Sj = density_estimation.SJoint(ll, pi=weights)
    Marginal = density_estimation.SMarginal(Sj)
    l = Marginal.mean()
    return l


def convert_gmm(init_filename):
    gmm = load_gmm(init_filename)
    mus = []
    covs = []
    weights = []
    for i in gmm:
        mus.append(i[1])
        covs.append(i[2])
        weights.append(i[0])
    weights = numpy.array(weights).reshape((len(weights), 1))
    return mus, covs, weights


D = numpy.load(
    "Data/GMM_data_4D.npy"
)

""" dens=numpy.load("/Users/manuelescobar/Documents/POLITO/2023-1/ML/library/Machine-Learning/labs/L10/Data/GMM_1D_3G_init_ll.npy")
S=numpy.array(ll_gaussian(D,mus,covs))
S=S+numpy.log(weights)
logdens = scipy.special.logsumexp(S, axis=0) """


""" m, c, w = EM(D,mus, covs, weights)
print(avg_ll(m,c,w)) """

mus, covs, weights = convert_gmm(
"Data/GMM_4D_3G_init.json"
)

m1, c1, w1 = LBG(D, 0.1, 2, psi=0.01)


m2, c2, w2 = EM(D, mus, covs, weights)


m3, c3, w3 = EM(D, mus, covs, weights, psi=0.01)
# print(avg_ll(m1,c1,w1))
print(avg_ll(m1, c1, w1))
print(avg_ll(m2, c2, w2))
print(avg_ll(m3, c3, w3))

""" m,c,w=convert_gmm("/Users/manuelescobar/Documents/POLITO/2023-1/ML/library/Machine-Learning/labs/L10/Data/GMM_1D_4G_EM_LBG.json")
print(m,c,w) """
# print(sol)
# print("---")
# print(m,c,w)
