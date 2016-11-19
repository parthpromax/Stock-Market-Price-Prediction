import numpy as np
from numpy import genfromtxt

train_data_file_name = 'train.csv'
test_data_file_name = 'test1.csv'
percentage=20

#Read traindata  
traindata = genfromtxt(train_data_file_name, delimiter=',')
traindata=np.delete(traindata, 0, 0)
trainx=np.delete(traindata, 0, 1)

trainx=np.delete(trainx, len(trainx[0])-1, 1)
trainx=np.delete(trainx, len(trainx[0])-1, 1)

X=trainx
trainy=traindata
for i in range (len(trainy[0])-2):
    trainy=np.delete(trainy, 0, 1)

YY=[]
for i in range(len(trainy)):
    if trainy[i][0]==1:
        YY.append(0)
    else:
        YY.append(1)
y=np.array(YY)

#cross-validation
import sklearn
from sklearn import svm
from sklearn.cross_validation import KFold

kf = KFold(len(X), n_folds=int(100.0/percentage))
print("Total folds : "+str(len(kf)))
sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,random_state=None)

acc=[]
count=0
for train_index, test_index in kf:
    print("\n\nFold no : "+str(count+1))
    count+=1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Gamma=3.1

    #kernel=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed'
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=Gamma, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)

    #fit model
    clf.fit(X_train, y_train)

    #predicate model
    predicated=clf.predict(X_test)
    print("Gamma : ",+Gamma)
    print("No of wrong predications : "+ str(sum(abs(predicated-y_test))) + " out of " + str(len(X_test)))
    temp=clf.score(X_test,y_test)
    print("Testing Accuracy : "+str(temp))
    acc.append(temp)

#fit original training data
clf.fit(X,y)
print("\n\nOverall  training accuracy : "+str(clf.score(X,y)))
print("\n\nOverall  testing accuracy : "+str(sum(acc)/len(acc)))