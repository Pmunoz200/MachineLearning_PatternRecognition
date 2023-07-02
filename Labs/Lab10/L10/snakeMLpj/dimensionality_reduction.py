from snakeMLpj.loads import loadData
from snakeMLpj.numpy_transformations import mcol
import numpy
import scipy
from snakeMLpj.visualization import scatter_attributeVSattribute

def PCA(x_train, m, x_test=[],flipped=False):
    mu=x_train.mean(1)
    DC=x_train-mcol(mu)
    C=numpy.dot(DC,DC.T)/x_train.shape[1]
    s, U = numpy.linalg.eigh(C)
    if flipped:
        P = -U[:, ::-1][:, 0:m]
    else:
        P = U[:, ::-1][:, 0:m]
    DPtrain=numpy.dot(P.T,x_train)
    if len(x_test)>0:
        DPtest=numpy.dot(P.T,x_test)
        return DPtrain, DPtest
    else:
        return DPtrain

def scatter_PCA(D,L,L_names,m,x_test=[],flipped=False, folder="", save=False, img_name=""):
    if len(x_test)>0:
        DPtrain, DPtest=PCA(D,m, flipped, x_test=x_test)
    else:
        DPtrain=PCA(D,m, flipped)
    features=[]
    for i in range(m):
        features.append("f"+str(i))
    scatter_attributeVSattribute(DPtrain,L,features,L_names,row_attributes=True,is_label_dict=True, name=img_name, folder=folder, save=save)
    if len(x_test)>0:
        return DPtrain, DPtest
    else:
        return DPtrain

def LDA(x_train, y_train, m, x_test=[], flipped=False):
    mu=x_train.mean(1)
    SW=numpy.zeros(x_train.shape[0])
    SB=numpy.zeros(x_train.shape[0])
    labels=numpy.unique(y_train)
    for i in labels:
        #SW
        data=x_train[:, y_train==labels[i]]
        muC=data.mean(1)
        DC=data-mcol(muC)
        CW=numpy.dot(DC,DC.T)
        SW=SW+CW
        #SB
        DM=mcol(muC)-mcol(mu)
        CB=numpy.dot(DM,DM.T)*data.shape[1]
        SB=SB+CB
    SW=SW/x_train.shape[1]
    SB=SB/x_train.shape[1]
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    DPtrain=numpy.dot(W.T,x_train)
    if len(x_test)>0:
        DPtest=numpy.dot(W.T,x_test)
        return DPtrain, DPtest
    else:
        return DPtrain

def scatter_LDA(D,L,L_names,m,x_test=[], flipped=False, folder="", save=False, img_name=""):
    if len(x_test)>0:
        DPtrain, DPtest=LDA(D,L, L_names,m, flipped)
    else:
        DPtrain=LDA(D,L, L_names,m, flipped)
    features=[]
    for i in range(m):
        features.append("f"+str(i))
    scatter_attributeVSattribute(DPtrain,L,features,L_names,row_attributes=True,is_label_dict=True, name=img_name, folder=folder, save=save)
    if len(x_test)>0:
        return DPtrain, DPtest
    else:
        return DPtrain

