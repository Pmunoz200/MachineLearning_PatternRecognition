import numpy
from scipy.optimize import fmin_l_bfgs_b
from snakeMLpj import validation, loads


class SVM():
    def __init__(self):
        self.params=[]
        self.predictions=[]
        self.metrics=[]
    
    def svm_obj(self, alpha, H):
        alpha=alpha.reshape(alpha.size,1)
        b=numpy.dot(alpha.T,H)
        c=numpy.dot(b,alpha)/2
        J=(c-alpha.sum())[0,0]
        gradient=numpy.dot(H,alpha)-1
        return J, gradient.flatten()
    
    def primal_solution(self, alpha, D, zi):
        a=numpy.multiply(alpha.reshape((alpha.size,1)),zi)
        w=numpy.multiply(a,D.T)
        return w.sum(axis=0)

    def extended_data_matrix(self, x, y, k):
        zi=2*y-1
        zi=zi.reshape((zi.size,1))
        zij=numpy.dot(zi, zi.T)
        newrow = numpy.ones(x.shape[1])*k
        D = numpy.vstack([x, newrow])
        return D, zij, zi

    def kernel_matrix(self, kernel, x1, x2, args=[]):
        if kernel=="Poly":
            IG= numpy.dot(x1.T,x2)+args[2]
            G=numpy.power(IG,args[3])+args[0]**2
        elif kernel=="RBF":
            a = x1[:, :, numpy.newaxis] - x2[:, numpy.newaxis, :]
            b = numpy.square(numpy.linalg.norm(a, axis=0))
            G=numpy.exp(-args[2]*b)+args[0]**2
        return G

    def obj_params(self, xtrain, ytrain, kernel, kernel_args=[]):    
        D, zij, zi=self.extended_data_matrix(xtrain,ytrain, kernel_args[0])
        if kernel=="linear":
            G= numpy.dot(D.T,D)
            H=numpy.multiply(zij, G)
        elif kernel=="Poly":
            G=self.kernel_matrix("Poly", xtrain,xtrain, args=kernel_args)
            H=numpy.multiply(zij, G)
        elif kernel=="RBF":
            G=self.kernel_matrix("RBF", xtrain,xtrain, args=kernel_args)
            H=numpy.multiply(zij, G)
        else:
            raise Exception("Kernel not implemented")
        return H, D, zi

    def train(self, xtrain, ytrain, factr=1, binary=True, models=["linear"]):
        self.params=[]
        self.predictions=[]
        self.metrics=[]
        self.models=models
        if binary:
            self.isBinary=True
        for i in self.models:
            if type(i) is tuple:
                model=i[0]
                model_args=i[1]
            else:
                print("Enter model arguments")
            x0 = numpy.ones(xtrain.shape[1])
            H, D, zi = self.obj_params(xtrain,ytrain, kernel=model, kernel_args=model_args)
            args=[H]
            bound=numpy.array([(0, model_args[1]) for i in range(xtrain.shape[1])])
            x, f, d= fmin_l_bfgs_b(self.svm_obj, x0, args=args, bounds=bound, factr=factr)
            if model=="linear":
                w=self.primal_solution(x, D, zi)
                self.params.append(w)
            else:
                a=x*zi.T
                self.params.append([a,xtrain])

    def evaluate(self, xtest, ytest, metric=["Accuracy"]):
        self.predictions=[]
        self.metrics=[]
        for i in range(len(self.params)):
            D, zij, zi=self.extended_data_matrix(xtest, ytest, self.models[i][1][0])
            if self.models[i][0]=="linear":
                score=numpy.dot(self.params[i].T,D)
            elif self.models[i][0]=="Poly" or self.models[i][0]=="RBF":
                k=self.kernel_matrix(self.models[i][0], self.params[i][1], xtest, self.models[i][1])
                score=numpy.dot(self.params[i][0],k)
            if self.isBinary:
                LP=[]
                for i in score.flatten():
                    if i>0:
                        LP.append(1)
                    else:
                        LP.append(0)
            else:
                LP=numpy.argmax(score, axis=0)
            self.predictions.append(LP)
            mod_metrics=metric.copy()
            for j in metric:
                    if type(j) is tuple:
                        if "minDCF" in j[0]:
                            args=metric[metric.index(j)][1].copy()
                            args.append(score.flatten())
                            mod_metrics[mod_metrics.index(j)]=("minDCF",args)
            m=validation.test_metrics(LP,ytest,metrics=mod_metrics)
            self.metrics.append(m)
        return self.predictions, self.metrics

    


        