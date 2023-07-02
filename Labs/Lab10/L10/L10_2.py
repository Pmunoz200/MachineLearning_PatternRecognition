from snakeMLpj import loads, gaussian
import numpy as np

D, L= loads.load_iris(row_attributes=True)
(x_train,y_train),(x_test,y_test)=loads.db_train_test_split(D,L)

gm=gaussian.generativeClassifier()
models=[('GMM',[0.1,0,0.01]),('GMM',[0.1,1,0.01]),('GMM',[0.1,2,0.01]),('GMM',[0.1,3,0.01]),('GMM',[0.1,4,0.01]),('tiedGMM',[0.1,0,0.01]),('tiedGMM',[0.1,1,0.01]),('tiedGMM',[0.1,2,0.01]),('tiedGMM',[0.1,3,0.01]),('tiedGMM',[0.1,4,0.01]),('diagGMM',[0.1,0,0.01]),('diagGMM',[0.1,1,0.01]),('diagGMM',[0.1,2,0.01]),('diagGMM',[0.1,3,0.01]),('diagGMM',[0.1,4,0.01])]
gm.train(x_train,y_train,models=models)
pred,m=gm.evaluate(x_test,y_test,metric=["Error"])
print(m)
