import numpy as np
from numpy import genfromtxt

#Declare Constants
no_of_input_perceptrons = 10
no_of_output_perceptrons = 2
no_of_perceptrons_in_hidden_layer1 = 28
no_of_perceptrons_in_hidden_layer2 = 28
no_of_iterations = 20
mode=2
learning_rate = 0.9
percentage = 20
train_data_file_name = 'train.csv'
test_data_file_name = 'test1.csv'

#Compute sigmoid activation function
def sigmoid(x,derivative=False):
    if(derivative==True):
        return x*(1-x)
    #Handle overflow due to exp function 
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]<-709:
                x[i][j]=-709
            elif x[i][j]>745:
                x[i][j]=745
    return 1/(1+np.exp(-x))

#Compute softmax activation function
def softmax(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]<-709:
                x[i][j]=-709
            elif x[i][j]>745:
                x[i][j]=745
    out=[]
    for i in range(len(x)):
        sum=0.0
        for k in range(len(x[0])):            
            sum+=np.exp(-x[i][k])
        for j in range(len(x[0])):
            out.append(np.exp(-x[i][j])/sum)
    out=(np.array(out)).reshape(len(x),len(x[0]))
    return out

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

y=trainy
expected_class=np.zeros((len(y),1))
for i in range(len(y)):
    if y[i][0]==1:
        expected_class[i]=0
    else:
        expected_class[i]=1

#Append column of 1s in input for bias
temp=np.zeros((len(X),no_of_input_perceptrons+1));
temp[:,:-1]=X
temp[:,len(X[0])]=[1]
X=temp

#cross-validation
import sklearn
from sklearn.cross_validation import KFold
kf = KFold(len(X), n_folds=int(100.0/percentage))
print("Total folds : "+str(len(kf)))
sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,random_state=None)
count=0
trainingacc=[]
testingacc=[]
i=0
for train_index, test_index in kf:
    if i==0:
        pass
        i=1
    else:
        break
    print("\n\nFold no : "+str(count+1))
    count+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    expected_class_train, expected_class_test = expected_class[train_index], expected_class[test_index]

    np.random.seed(40)
    
    #Initialize weight matrices randomly with mean 0
    syn0 = 2*np.random.random((no_of_input_perceptrons+1,no_of_perceptrons_in_hidden_layer1)) - 1
    syn1 = 2*np.random.random((no_of_perceptrons_in_hidden_layer1+1,no_of_perceptrons_in_hidden_layer2)) - 1
    syn2 = 2*np.random.random((no_of_perceptrons_in_hidden_layer2+1,no_of_output_perceptrons)) - 1
    
    for iter in range(no_of_iterations):
        errorr1=0
        for i in range(len(X_train)):
            
    		#Forward propagation
            layer0_output = X_train[i].reshape(1,no_of_input_perceptrons+1)
    
            layer1_output = sigmoid((np.dot(layer0_output,syn0)).reshape(1,no_of_perceptrons_in_hidden_layer1))
    		
            temp=np.zeros((1,len(layer1_output[0])+1));
            temp[:,:-1]=layer1_output
            temp[:,len(layer1_output[0])]=[1]
            layer1_output=temp
    
            layer2_output = sigmoid((np.dot(layer1_output,syn1)).reshape(1,no_of_perceptrons_in_hidden_layer2))
            
            temp=np.zeros((1,len(layer2_output[0])+1));
            temp[:,:-1]=layer2_output
            temp[:,len(layer2_output[0])]=[1]
            layer2_output=temp
    
            layer3_output = (softmax((np.dot(layer2_output,syn2)).reshape(1,no_of_output_perceptrons))).reshape(1,no_of_output_perceptrons)
            
            for l in range(len(layer3_output)):
                max=layer3_output[l][0]
                ans=0
                for j in range(1,len(layer3_output[0])):
                    if layer3_output[l][j]>max:
                        max=layer3_output[l][j]
                        ans=j
            if y_train[i][ans]!=1:
                errorr1+=1
            
    		#Back propagation
         	#Compute error and multiply it with derivative 	
            layer3_delta = np.multiply( (layer3_output-(y_train[i,:]).reshape(1,no_of_output_perceptrons)), sigmoid(layer3_output,derivative=True).reshape(1,no_of_output_perceptrons) )
            
            layer2_delta = np.multiply(np.dot(layer3_delta,syn2.T),(sigmoid(layer2_output,derivative=True)).reshape(1,no_of_perceptrons_in_hidden_layer2+1))
            
            layer1_delta =  np.multiply(np.dot(layer2_delta[:,:-1],syn1.T),(sigmoid(layer1_output,derivative=True)).reshape(1,no_of_perceptrons_in_hidden_layer1+1))
    		
    		#Update weights
            syn2 += (learning_rate)*(np.dot(layer2_output.T,layer3_delta))
            syn1 += (learning_rate)*(np.dot(layer1_output.T,layer2_delta[:,:-1]))
            syn0 += (learning_rate)*(np.dot(layer0_output.T,layer1_delta[:,:-1]))
        
        if iter%mode==0:
            print()
            print("After "+str(iter)+" iterations");
            print ("No of wrong predications :" + str(errorr1) + " out of " + str(len(y_train)))
            print ("Training accuracy :" + str((len(X_train)-errorr1)/len(X_train)))
    
    layer1_predication = sigmoid(np.dot(X_train,syn0))
    
    temp=np.zeros((len(layer1_predication),len(layer1_predication[0])+1));
    temp[:,:-1]=layer1_predication
    temp[:,len(layer1_predication[0])]=[1]
    layer1_predication=temp
    
    layer2_predication = sigmoid(np.dot(layer1_predication,syn1))
    
    temp=np.zeros((len(layer2_predication),len(layer2_predication[0])+1));
    temp[:,:-1]=layer2_predication
    temp[:,len(layer2_predication[0])]=[1]
    layer2_predication=temp
    
    layer3_predication = softmax(np.dot(layer2_predication,syn2))
    
    #Write predicated class to output file in required format 
    predicated_class=[]
    for i in range(len(layer3_predication)):
        max=layer3_predication[i][0]
        temp=0
        for j in range(1,len(layer3_predication[0])):
            if layer3_predication[i][j]>max:
                max=layer3_predication[i][j]
                temp=j
        predicated_class.append(temp)        
    
    predicated_class=(np.array(predicated_class)).reshape(len(predicated_class),1)
    temp=(len(y_train)-sum(sum(abs(expected_class_train-predicated_class))))/len(y_train)
    print("\n\nFinal Training accuracy : "+str(temp))
    trainingacc.append(temp)
    
    #Predicate class for test data
    layer1_predication = sigmoid(np.dot(X_test,syn0))
    
    temp=np.zeros((len(layer1_predication),len(layer1_predication[0])+1));
    temp[:,:-1]=layer1_predication
    temp[:,len(layer1_predication[0])]=[1]
    layer1_predication=temp
    
    layer2_predication = sigmoid(np.dot(layer1_predication,syn1))
    
    temp=np.zeros((len(layer2_predication),len(layer2_predication[0])+1));
    temp[:,:-1]=layer2_predication
    temp[:,len(layer2_predication[0])]=[1]
    layer2_predication=temp  
    
    layer3_predication = softmax(np.dot(layer2_predication,syn2))
    
    #Write predicated class to output file in required format 
    predicated_class=[]
    for i in range(len(layer3_predication)):
        max=layer3_predication[i][0]
        temp=0
        for j in range(1,len(layer3_predication[0])):
            if layer3_predication[i][j]>max:
                max=layer3_predication[i][j]
                temp=j
        predicated_class.append(temp)        
    
    predicated_class=(np.array(predicated_class)).reshape(len(predicated_class),1)
    temp=(len(y_test)-sum(sum(abs(expected_class_test-predicated_class))))/len(y_test)
    print("Final Testing accuracy : "+str(temp))
    testingacc.append(temp)

print("\n\nOverall training accuracy : "+str(sum(trainingacc)/len(trainingacc)))
print("Overall testing accuracy : "+str(sum(testingacc)/len(testingacc)))