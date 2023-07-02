import numpy
from scipy.optimize import fmin_l_bfgs_b
from snakeMLpj import validation


class logisticRegression():
    def __init__(self):
        self.params=None
        self.predictions=[]
        self.metrics=None

    def train(self, xtrain, ytrain, metric=["Accuracy"], l=0.001, binary=False, models=None):
        self.params=[]
        self.predictions=[]
        self.metrics=[]
        if binary:
            self.isBinary=True
            self.binary_logRegression(xtrain, ytrain, l)
        else:
            self.isBinary=False
            self.logRegression(xtrain, ytrain, l)

    def evaluate(self, xtest, ytest, metric=["Accuracy"]):
        self.predictions=[]
        self.metrics=None
        score=numpy.dot(self.params[0].T,xtest)+self.params[1]
        if self.isBinary:
            LP=[]
            for i in score:
                if i>0:
                    LP.append(1)
                else:
                    LP.append(0)
        else:
            LP=numpy.argmax(score, axis=0)
        self.predictions=LP
        mod_metrics=metric.copy()
        for j in metric:
                if type(j) is tuple:
                    if "minDCF" in j[0]:
                        args=metric[metric.index(j)][1].copy()
                        args.append(score)
                        mod_metrics[mod_metrics.index(j)]=("minDCF",args)
        m=validation.test_metrics(self.predictions,ytest,metrics=mod_metrics)
        self.metrics=[m]
        return self.predictions, self.metrics
    
    def binary_logreg_obj(self, v, xtrain, ytrain, l):
        w, b = v[0:-1], v[-1]
        zi=2*ytrain-1
        p0=numpy.linalg.norm(w)
        r0=(l/2)*numpy.power((p0),2)
        p1=(numpy.dot(w.T,xtrain) + b)
        p2=numpy.multiply(-zi,p1)
        p3=numpy.logaddexp(0, p2)
        r1=p3.sum(axis=0)/xtrain.shape[1]
        return r0+r1

    def logreg_obj(self, v, xtrain, ytrain, l, k):
        w, b = v[0:-k], v[-k::]
        w=w.reshape((xtrain.shape[0], k))
        b=b.reshape((k,1))
        S = numpy.dot(w.T, xtrain) + b
        Sexp=numpy.exp(S)
        Ssum=numpy.sum(Sexp,axis=0)
        Slog=numpy.log(Ssum)
        ylog=S-Slog.reshape((1,Slog.shape[0]))
        Tki=numpy.zeros((k,ytrain.shape[0]))
        for i in range(ytrain.shape[0]):
            Tki[ytrain[i],i]=1
        TY=numpy.multiply(ylog,Tki)
        p0=numpy.linalg.norm(w)
        r0=(l/2)*numpy.power((p0),2)
        return r0-numpy.sum(TY)/xtrain.shape[1]

    def binary_logRegression(self,xtrain, ytrain, l):
        x0 = numpy.zeros(xtrain.shape[0] + 1)
        args=(xtrain, ytrain, l)
        x, f, d= fmin_l_bfgs_b(self.binary_logreg_obj, x0, args=args, approx_grad=True)
        wr, br = x[0:-1], x[-1]
        self.params=(wr,br)

    def logRegression(self, xtrain, ytrain, l):
        k=numpy.unique(ytrain).shape[0]
        x0 = numpy.zeros((xtrain.shape[0] + 1)*k)
        args=(xtrain, ytrain, l, k)
        x, f, d= fmin_l_bfgs_b(self.logreg_obj, x0, args=args, approx_grad=True)
        x=x.reshape((xtrain.shape[0] + 1,k))
        wr, br = x[0:-1,:], x[-1,:]
        br=br.reshape((br.shape[0],1))
        self.params=(wr,br)
        