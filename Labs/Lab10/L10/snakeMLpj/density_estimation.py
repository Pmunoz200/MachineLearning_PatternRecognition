import numpy
import matplotlib.pyplot as plt
import math
import scipy
from snakeMLpj.numpy_transformations import mcol, mrow, mean_cov

def SJoint(ll, pi=None, logarithmic=True):
    ll=numpy.array(ll)
    classes=numpy.array(ll).shape[0]
    if pi is None:
        pi=numpy.full((classes,1),1/classes)
    SJoint=numpy.array(ll)
    pi=numpy.array(pi)
    if logarithmic:
        SJoint=SJoint+numpy.log(pi.reshape((-1,1)))
    else:
        SJoint=SJoint*(pi)
    return SJoint

def SPost(SJoint, SMarginal, logarithmic=True, exp=True):
    if logarithmic:
        logSPost = SJoint - SMarginal
        if exp:
            return numpy.exp(logSPost)
        else:
            return logSPost
    else:
        return SJoint/SMarginal

def SMarginal(SJoint, logarithmic=True):
    if logarithmic:
        return  mrow(scipy.special.logsumexp(SJoint, axis=0))
    else:
        return mrow(SJoint.sum(0))

def SPost_from_ll(ll, pi, log=True):
    SJ=SJoint(ll, pi=pi, logarithmic=log)
    Marginal=SMarginal(SJ, logarithmic=log)
    Posterior=SPost(SJ,Marginal, logarithmic=log)
    return Posterior

def estimation(ll, log=True, pi=None):
    #ll=loglikelihood_ll(xtest, mu,cov, log=False)
    SJ=SJoint(ll, pi=pi, logarithmic=log)
    Marginal=SMarginal(SJ, logarithmic=log)
    Posterior=SPost(SJ,Marginal, logarithmic=log)
    pred=numpy.argmax(Posterior, axis=0)
    return pred
    
def logpdf_GAU_ND(x, mu=False, C=False, exp=False):
    if (not mu.any() and not C.any()):
        mu, C= mean_cov(x)
    C_inv=numpy.linalg.inv(C)
    det = numpy.linalg.slogdet(C)[1]
    M=x.shape[0]
    log_pi=math.log(2*math.pi)
    #ab=(-M*log_pi-det)/2
    x_mu=numpy.subtract(x,mu)
    r1=numpy.dot(x_mu.T,C_inv)
    r2=numpy.diagonal(numpy.dot(r1,x_mu))
    result=(-M*log_pi-det-r2)/2
    if exp:
        return numpy.exp(result)
    else:
        return result

def logpdf_GAU_ND_error(Solution, Result):
    print("Error: ",numpy.abs(Solution - Result).max())
 
def logpdf_GAU_ND_visualization(Data, result):
    plt.figure()
    plt.plot(Data.ravel(), numpy.exp(result))
    plt.show()

def loglikelihood_visualization(Data, XPlot, Result):
    plt.figure()
    plt.hist(Data.ravel(), bins=50, density=True)
    plt.plot(XPlot.ravel(), numpy.exp(Result))
    plt.show()

def loglikelihood(x, m_ML=False, C_ML=False, return_log_density=False, visualize=numpy.array([])):
    if (not m_ML and not C_ML):
        """ mu=mcol(numpy.mean(x,axis=1))
        c=numpy.cov(x)
        if c.size==1:
            c=numpy.reshape(numpy.array(c),(c.size,-1)) """
        gau=logpdf_GAU_ND(x)
        if visualize.size>0:
            loglikelihood_visualization(x, visualize, logpdf_GAU_ND(mrow(visualize),mu,c))
    else:
        gau=logpdf_GAU_ND(x,m_ML,C_ML)
        if visualize.size>0:
            loglikelihood_visualization(x, visualize, logpdf_GAU_ND(mrow(visualize),m_ML,C_ML))
    result=numpy.sum(gau)
    if return_log_density:
        return result, gau
    else:
        return result