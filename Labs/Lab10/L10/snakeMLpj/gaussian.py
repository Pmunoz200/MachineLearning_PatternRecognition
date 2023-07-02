#---------MVG----------
import scipy
import numpy
from snakeMLpj.validation import test_metrics
import math
from snakeMLpj.numpy_transformations import mean_cov, mrow, wc_cov, mean
import numpy
from snakeMLpj import density_estimation, numpy_transformations

class generativeClassifier():
    def __init__(self):
        self.params=[]
        self.predictions=[]
        self.metrics=[]

    def train(self, x_train, y_train, models=['MVG','logMVG','NBG','logNBG','TiedMVG','logTiedMVG','TiedNBG','logTiedNBG','GMM'], Pc=False, show_accuracy=False):
        self.params=[]
        self.predictions=[]
        self.metrics=[]
        self.models=models
        classes=numpy.unique(y_train)
        if not Pc:
            self.Pc=1/classes.size
        for modelf in self.models:
            if type(modelf) is tuple:
                model=modelf[0]
                args=modelf[1]
            else:
                model=modelf
            match(model):
                case("MVG"):
                    isLog=False
                    isTied=False
                    mus, covs = self.MVG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("logMVG"):
                    isLog=True
                    isTied=False
                    mus, covs = self.MVG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("TiedMVG"):
                    isLog=False
                    isTied=True
                    mus, covs = self.MVG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("logTiedMVG"):
                    isLog=True
                    isTied=True
                    mus, covs = self.MVG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("NBG"):
                    isLog=False
                    isTied=False
                    mus, covs = self.NBG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("logNBG"):
                    isLog=True
                    isTied=False
                    mus, covs = self.NBG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("TiedNBG"):
                    isLog=False
                    isTied=True
                    mus, covs = self.NBG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("logTiedNBG"):
                    isLog=True
                    isTied=True
                    mus, covs = self.NBG(x_train,y_train,isTied)
                    self.params.append((mus,covs, isLog))
                case("GMM"):
                    isLog=True
                    isTied=True
                    mus, covs, weights = self.LBGclasses(x_train, y_train,alpha=args[0],num_splits=args[1],psi=args[2])
                    self.params.append((mus,covs, isLog,weights))
                case("tiedGMM"):
                    isLog=True
                    isTied=True
                    mus, covs, weights = self.LBGclasses(x_train, y_train, alpha=args[0],num_splits=args[1],psi=args[2], tied=True)
                    self.params.append((mus,covs, isLog,weights))
                case("diagGMM"):
                    isLog=True
                    isTied=True
                    mus, covs, weights = self.LBGclasses(x_train,y_train, alpha=args[0],num_splits=args[1],psi=args[2],diag=True)
                    self.params.append((mus,covs, isLog,weights))
        if show_accuracy:
            print("---Training Accuracy---")
            pred, metric = self.evaluate(x_train, y_train, metric=["Accuracy"], show_results=True)

    def train_evaluate(self, x_train, y_train,  x_test, y_test, models=['MVG','logMVG','NBG','logNBG','TiedMVG','logTiedMVG','TiedNBG','logTiedNBG'], Pc=False):
        self.train(x_train,y_train,models,Pc)
        self.evaluate(x_test,y_test)

    def evaluate(self, x_test, y_test, metric=["Accuracy"], show_results=False):
        self.predictions=[]
        self.metrics=[]

        for i in range(len(self.params)):
            mod_metrics=metric.copy()
            if len(self.params[i])>3:
                score=[]
                for h in range(len(self.params[i][0])):
                    ll=self.ll_gaussian(x_test,self.params[i][0][h],self.params[i][1][h], log=self.params[i][2])
                    Sj=density_estimation.SJoint(ll, pi=self.params[i][3][h])
                    Marginal=density_estimation.SMarginal(Sj)
                    score.append(Marginal)
                score=numpy.array(score)
                score=score.reshape((len(self.params[i][0]),-1))
                pred=numpy.argmax(score, axis=0)
            else:
                score=self.ll_gaussian(x_test,self.params[i][0],self.params[i][1], log=self.params[i][2])
                pred =density_estimation.estimation(score, log=self.params[i][2])
            for j in metric:
                if type(j) is tuple:
                    if "minDCF" in j[0]:
                        args=metric[metric.index(j)][1].copy()
                        args.append(score)
                        mod_metrics[mod_metrics.index(j)]=("minDCF",args)
            err=test_metrics(pred,y_test, mod_metrics)
            self.predictions.append(pred)
            self.metrics.append(err)
            if show_results:
                print("Model:", self.models[i],"| ", "Accuracy:", metric[0])
        return self.predictions, self.metrics

    def ll_gaussian(self, xtest, mu, C, log=True):
        score=[]
        for i in range(len(C)):
            C_inv=numpy.linalg.inv(C[i])
            det = numpy.linalg.slogdet(C[i])[1]
            M=xtest.shape[0]
            log_pi=math.log(2*math.pi)
            x_mu=numpy.subtract(xtest,mu[i].reshape((-1,1)))
            r1=numpy.dot(x_mu.T,C_inv)
            r2=numpy.diagonal(numpy.dot(r1,x_mu))
            result=(-M*log_pi-det-r2)/2
            if log:
                score.append(result)
            else:
                score.append(numpy.exp(result))
        return score

    def EM(self, D, mus,covs,weights, threshold=10**(-6), psi=None, diagonal=False, tied=False):
        l_old=-numpy.inf
        lg=numpy.inf
        g=len(mus)
        mu_1=numpy.array(mus)
        cov_1=numpy.array(covs)
        w_1=numpy.array(weights)
        ll=self.ll_gaussian(D,mu_1,cov_1)
        Sj=density_estimation.SJoint(ll, pi=w_1)
        Marginal=density_estimation.SMarginal(Sj)
        r=density_estimation.SPost(Sj,Marginal, exp=True)
        Zg=r.sum(axis=1).reshape((-1,1))
        #print(cov_1.shape)
        if diagonal:
            covnew=[]
            for k in range(g):
                    Sigma_g = cov_1[k,:,:] * numpy.eye(cov_1.shape[1])
                    covnew.append(Sigma_g)
            cov_1=covnew
        if tied:
            Sigma_g=numpy.zeros((g,cov_1.shape[1],cov_1.shape[1]))
            for k in range(g):
                    Sigma_g += Zg[k,:]*cov_1[k,:,:]
            cov_1=Sigma_g/D.shape[1]
        if psi:
            covnew=[]
            for k in range(g):
                    cov_1=numpy.array(cov_1)
                    U, s, _ = numpy.linalg.svd(cov_1[k,:,:])
                    s[s<psi] = psi
                    covnew.append(numpy.dot(U, numpy_transformations.mcol(s)*U.T))
            cov_1=covnew
        while lg>=threshold:
            ll=self.ll_gaussian(D,mu_1,cov_1)
            Sj=density_estimation.SJoint(ll, pi=w_1)
            Marginal=density_estimation.SMarginal(Sj)
            r=density_estimation.SPost(Sj,Marginal, exp=True) 
            Fg=numpy.dot(r,D.T)
            Zg=r.sum(axis=1).reshape((-1,1))
            mu_1=Fg/Zg
            w_1=Zg
            w_1=(w_1/w_1.sum()).reshape((-1,1))
            Sg=[]
            cov_1=[]
            b=[]
            for i in range(g):
                psg=numpy.zeros((D.shape[0],D.shape[0]))
                for j in range(D.shape[1]):
                    y=r[i,j]
                    xi=D[:,j].reshape((-1,1))
                    xii=numpy.dot(xi, xi.T)
                    psg+=y*xii
                Sg.append(psg)
                b.append(numpy.dot(mu_1[i,:].reshape((-1,1)),mu_1[i,:].reshape((1,-1))))
            Sg=numpy.array(Sg)
            a=Sg/Zg.reshape((-1,1,1))
            b=numpy.array(b)
            cov_1=a-b
            if diagonal:
                covnew=[]
                for k in range(g):
                        Sigma_g = cov_1[k,:,:] * numpy.eye(cov_1.shape[1])
                        covnew.append(Sigma_g)
                cov_1=covnew
            if tied:
                Sigma_g=numpy.zeros((g,cov_1.shape[1],cov_1.shape[1]))
                for k in range(g):
                        Sigma_g += Zg[k,:]*cov_1[k,:,:]
                cov_1=Sigma_g/D.shape[1]
            if psi:
                covnew=[]
                for k in range(g):
                        cov_1=numpy.array(cov_1)
                        U, s, _ = numpy.linalg.svd(cov_1[k,:,:])
                        s[s<psi] = psi
                        covnew.append(numpy.dot(U, numpy_transformations.mcol(s)*U.T))
                cov_1=covnew
            l=Marginal.mean()
            lg=l-l_old
            l_old=l
            #print("loss:",lg)
        return mu_1, cov_1, w_1

    def LBG(self, D, alpha=0.1, num_splits=2, psi=None, diag=False, tied=False):
        mu, C =numpy_transformations.mean_cov(D)
        mu=numpy.array([mu])
        C=numpy.array([C])
        w=numpy.array([1]).reshape((-1,1,1))
        #print(mu.shape, C.shape, w.shape)
        for j in range(num_splits):
            mu_split=[]
            C_split=[]
            w_split=[]
            for i in range(len(mu)):
                U, s, Vh = numpy.linalg.svd(C[i])
                d = U[:, 0:1] * s[0]**0.5 * alpha
                mu_split.append(mu[i,:,:]-d)
                mu_split.append(mu[i,:,:]+d)
                C_split.append(C[i,:,:])
                C_split.append(C[i,:,:])
                w_split.append(w[i,:,:]/2)
                w_split.append(w[i,:,:]/2)
            mu,C,w=self.EM(D,mu_split,C_split,w_split, psi=psi, tied=tied, diagonal=diag)
            mu=numpy.array(mu).reshape((-1,D.shape[0],1))
            C=numpy.array(C)
            w=numpy.array(w).reshape((-1,1,1))
            #print(mu.shape, C.shape, w.shape)
        return mu, C, w
    
    def LBGclasses(self,xtrain,ytrain,  alpha=0.1, num_splits=2, psi=None, diag=False, tied=False):
        mus=[]
        covs=[]
        weights=[]
        for i in numpy.unique(ytrain):
            mu, C, w= self.LBG(xtrain[:,ytrain==i],alpha,num_splits,psi,diag,tied)
            mus.append(mu)
            covs.append(C)
            weights.append(w)
        return mus, covs, weights

    def MVG(self,x_train, y_train, tied=False):
        mus=[]
        covs=[]
        if tied:
            C=wc_cov(x_train,y_train)
        for i in numpy.unique(y_train):
            if tied:
                mu=mean(x_train[:,y_train==i])
            else:
                mu, C= mean_cov(x_train[:,y_train==i])
            mus.append(mu)
            covs.append(C)
        return mus, covs

    def NBG(self,x_train, y_train, tied=False):
        mus=[]
        covs=[]
        if tied:
            C=wc_cov(x_train,y_train)
            C=C*numpy.identity(C.shape[0])
        for i in numpy.unique(y_train):
            if tied:
                mu=mean(x_train[:,y_train==i])
            else:
                mu, C= mean_cov(x_train[:,y_train==i])
                C=C*numpy.identity(C.shape[0])
            mus.append(mu)
            covs.append(C)
        return mus, covs