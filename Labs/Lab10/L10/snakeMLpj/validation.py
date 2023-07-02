from matplotlib import pyplot as plt
import numpy
from snakeMLpj import density_estimation
from snakeMLpj.dimensionality_reduction import LDA, PCA

def matrix_max_error(m1,m2):
    print("Error: ",numpy.abs(m1 - m2).max())

def accuracy(predicted, test):
    correct=0
    for i in range(len(predicted)):
        if predicted[i]==test[i]:
            correct+=1
    #print("Accuracy: ",(correct/len(predicted))*100, "%")
    return correct/len(predicted)*100

def error(predicted, test):
    wrong=0
    for i in range(len(predicted)):
        if predicted[i]!=test[i]:
            wrong+=1
    #print("Error: ",(wrong/len(predicted))*100, "%")
    return wrong/len(predicted)*100

def confusion_matrix(predicted, y_test, k=None):
    try:
        k1=numpy.max(numpy.unique(y_test))
        k2=numpy.max(numpy.unique(predicted))
        k=max(int(k1),int(k2))
        if k<1:
            k=1
        cm=numpy.zeros((k+1,k+1))
        for i in range(len(predicted)):
            cm[int(predicted[i]),int(y_test[i])]=cm[int(predicted[i]),int(y_test[i])]+1
        return cm
    except:
        print("Must specify k (num of classes)")


def optimalBayesDecision(ll,pi, C, isll_binaryratio=True, t=None):
    if isll_binaryratio:
        pred=numpy.zeros(ll.size)
        if not t:
            t=-numpy.log(pi[1]*C[0][1])/((1-pi[1])*C[1][0])
        pred[ll>(t)]=1
    else:
        pi=numpy.array(pi)
        pi=pi.reshape(pi.shape[0],1)
        post=density_estimation.estimate(ll, pi)
        cp=numpy.dot(C,post)
        pred=numpy.argmin(cp,axis=0)
    return pred

def DCF( pi, C, conf_matrix=None, pred_ytest=None, normalized=True):
    if conf_matrix is None:
        if pred_ytest:
            conf_matrix=confusion_matrix(pred_ytest[0], pred_ytest[1])
        else:
            raise Exception("Enter conf_matrix or pred_ytest=(pred,ytest)")
    if len(C)==2:
        FNR=conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
        FPR=conf_matrix[1][0]/(conf_matrix[0][0]+conf_matrix[1][0])
        dcf=pi[1]*C[0][1]*FNR+pi[0]*C[1][0]*FPR
        if normalized:
            return dcf/min(pi[1]*C[0][1],pi[0]*C[1][0])
        else:
            return dcf
    cms=numpy.array(conf_matrix).sum(axis=0)
    r=conf_matrix/cms
    rc=numpy.multiply(r.T, C)
    rc_s=rc.sum(axis=1)
    rcs=numpy.multiply(rc_s,pi).sum(axis=0)
    if normalized:
        cp=numpy.min(numpy.dot(C, pi))
        return rcs/cp
    else:
        return rcs
    
def minDCF(llr, pi, C, y_test):
    ll=numpy.array(llr)
    if ll.ndim>1:
        llr= ll[1,:]-ll[0,:]
    thresholds=[-numpy.inf, numpy.inf]
    thresholds.extend(llr)
    thresholds=numpy.sort(thresholds)
    dcfs=[]
    for i in thresholds:
        pred=optimalBayesDecision(llr, pi, C, t=i)
        dcfs.append(DCF(pi, C, pred_ytest=(pred, y_test)))
    return min(dcfs)

def binary_DCF_minDCF(ll, pi, C, y_test):
    pred=optimalBayesDecision(ll, pi, C)
    cm=confusion_matrix(pred, y_test)
    dcf=DCF(cm, pi, C)
    mindcf=minDCF(ll, pi, C, y_test)
    return dcf, mindcf

def ROC(ll, pi, C, y_test):
    thresholds=[-numpy.inf, numpy.inf]
    thresholds.extend(ll)
    thresholds=numpy.sort(thresholds)
    x=[]
    y=[]
    for i in thresholds:
        pred=optimalBayesDecision(ll, pi, C, t=i)
        cm=confusion_matrix(pred, y_test)
        x.append(cm[1][0])
        y.append(cm[1][1])
    #plot ROC curve
    plt.plot(x/max(x),y/max(y))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

def bayes_error_plot(ll,y_test, precision=0.25, effPriorLB=-3, effPriorUB=3):
    effPriorLogOdds=numpy.arange(effPriorLB,effPriorUB,step=precision)
    c=[[0,1],[1,0]]
    dcf=[]
    mindcf=[]
    for p in effPriorLogOdds:
        eff=1/(1+numpy.exp(-p))
        effPi=[1-eff, eff]
        pred=optimalBayesDecision(ll, effPi, c)
        cm=confusion_matrix(pred, y_test)
        dcf.append(DCF(cm, effPi, c))
        mindcf.append(minDCF(ll,effPi,c,y_test))
    plt.plot(effPriorLogOdds, dcf, label="DCF", color="r")
    plt.plot(effPriorLogOdds, mindcf, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

def test_metrics(predicted, y_test, metrics):
    testm=[]
    for i in metrics:
        if type(i) is tuple:
            metric=i[0]
        else:
            metric=i
        m=0
        if metric=="Accuracy":
            m=accuracy(predicted, y_test)
        elif metric=="Error":
            m=error(predicted, y_test)
        elif metric=="DCF":
            wp=i[1][0]
            if wp:
                m=DCF(pi=wp[0], C=wp[1], pred_ytest=(predicted, y_test))
            else:
                raise Exception("Missing working point")
        elif metric=="minDCF":
            wp=i[1][0]
            ll=i[1][1]
            if wp and len(ll)>0:
                m=minDCF(ll, pi=wp[0], C=wp[1], y_test=y_test)
            else:
                raise Exception("Missing working point or ll")
        testm.append(m)
    return testm

def kfold(D, L, model, submodels, k=1, PCAm=None, LDAm=None, PCAandLDA=False, metric=["Accuracy"], seed=0):
        if not k>0:
            print("Invalid K ---> k=1: LOO or k>1")
            return 0
        indexes = numpy.arange(D.shape[1])
        errors=[]
        if k==1:
            for i in range(D.shape[1]):
                y_train = L[indexes!=i]
                y_test = L[indexes==i]
                x_train = D[:, indexes!=i]
                x_test = D[:, indexes==i]
                if PCAm:
                    x_train, x_test = PCA(x_train, PCAm, x_test=x_test)
                    if LDAm:
                        x_train, x_test = LDA(x_train, y_train, LDAm, x_test=x_test)
                elif LDAm:
                    x_train, x_test = LDA(x_train, y_train, LDAm, x_test=x_test)
                model.train(x_train,y_train, models=submodels)
                pred, m = model.evaluate(x_test, y_test, metric=metric)
                errors.append(m)
        else:
            numpy.random.seed(seed)
            idx = numpy.random.permutation(D.shape[1])
            folds=[]
            for i in range(k):
                folds.append(idx[int(i*(D.shape[1]/k)):int((i+1)*(D.shape[1]/k))])
            for i in range(len(folds)):
                c = numpy.in1d(indexes, folds[i])
                cinv=numpy.invert(c)
                x_train = D[:, cinv]
                x_test = D[:, c]
                y_train = L[cinv]
                y_test = L[c]
                if PCAm:
                    x_train, x_test = PCA(x_train, PCAm, x_test=x_test)
                    if LDAm:
                        x_train, x_test = LDA(x_train, y_train, LDAm, x_test=x_test)
                elif LDAm:
                    x_train, x_test = LDA(x_train, y_train, LDAm, x_test=x_test)
                model.train(x_train,y_train, models=submodels)
                pred, m = model.evaluate(x_test, y_test, metric=metric)
                errors.append(m)
        errors=numpy.array(errors).mean(axis=0)
        return errors