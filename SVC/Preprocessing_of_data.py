import numpy as np
import pandas as pd
from numpy import genfromtxt

Days=10
def minn(X):
    min=X[0];
    for i in range(1,len(X)):
        if min>X[i]:
            min=X[i]
    return min
        
def maxx(X):
    max=X[0];
    for i in range(1,len(X)):
        if max<X[i]:
            max=X[i]
    return max

X = genfromtxt('nik225.csv', delimiter=',',usecols=range(1,5),skip_header=True)
y = X[:,3]

SMAA=np.zeros((len(X)-(Days-1),1))
for i in range((Days-1),len(X)):
    SMAA[i-(Days-1)]=np.average(X[i-(Days-1):i+1,3])
#print(SMAA) #1

WMA=np.zeros((len(X)-(Days-1),1))
for i in range((Days-1),len(X)):
    sum1=0.0
    for j in range(i-(Days-1),i+1):
        sum1+=(Days-i+j)*X[j][3]                    
    sum1/=(Days*(Days+1))/2
    WMA[i-(Days-1)]=sum1
#print(WMA) #2


Momentum=np.zeros((len(X)-(Days-1),1))
for i in range((Days-1),len(X)):
    Momentum[i-(Days-1)]=X[i][3]-X[i-(Days-1)][3]
#print(Momentum) #3

StochasticK=np.zeros((len(X)-(Days-1),1))
for i in range((Days-1),len(X)):
    StochasticK[i-(Days-1)]=(X[i][3]-minn(X[i-(Days-1):i+1,2]))*100.0/(maxx(X[i-(Days-1):i+1,1])-minn(X[i-(Days-1):i+1,2]))
#print(StochasticK) #4

StochasticD=np.zeros((len(StochasticK)-(Days-1),1))
for i in range((Days-1),len(StochasticK)):
    sum1=0.0
    for j in range(i-(Days-1),i+1):
        sum1+=StochasticK[j]                 
    sum1/=Days
    StochasticD[i-(Days-1)]=sum1

#print(StochasticD) #5

UP=np.zeros((len(X)-1,1))
DW=np.zeros((len(X)-1,1))
for i in range(1,len(X)):
    temp=X[i][3]-X[i-1][3]
    if temp<0:
        UP[i-1]=0
        DW[i-1]=-temp
    else:
        UP[i-1]=temp
        DW[i-1]=0
#print(UP)
#print(DW)


RSI=np.zeros((len(UP)-(Days-1),1))
for i in range((Days-1),len(UP)):
        RSI[i-(Days-1)]=100-(100/(1+(np.average(UP[i-(Days-1):i+1])/np.average(DW[i-(Days-1):i+1]))))
    #print(RSI) #6

EMA12=np.zeros((len(X),1))
EMA12[0]=X[0][3]
for i in range(1,len(EMA12)):
    EMA12[i]=EMA12[i-1]+( (2/(1+12)) * (X[i][3]-EMA12[i-1]) )

EMA26=np.zeros((len(X),1))
EMA26[0]=X[0][3]
for i in range(1,len(EMA26)):
    EMA26[i]=EMA26[i-1]+( (2/(1+26)) * (X[i][3]-EMA26[i-1]) )

DIFF=EMA12-EMA26

MACD=np.zeros((len(X),1))
MACD[0]=DIFF[0]
for i in range(1,len(EMA26)):
    MACD[i]=MACD[i-1]+( (2/(len(MACD)+1)) * (DIFF[i]-MACD[i-1]) )
#print(MACD) #7


LWR=np.zeros((len(X),1))
for i in range(len(X)):
    LWR[i]=(X[i][1]-X[i][3])/(X[i][1]-X[i][2]) if (X[i][1]-X[i][2])!=0 else 0.00000000000001
#print(LWR) #8

ADO=np.zeros((len(X)-1,1))
for i in range(1,len(X)):
    ADO[i-1]=(X[i][1]-X[i-1][3])/(X[i][1]-X[i][2]) if (X[i][1]-X[i][2])!=0 else 0.00000000000001
#print(ADO) #9

M=np.zeros((len(X),1))
for i in range(len(X)):
    M[i]=(X[i][1]+X[i][2]+X[i][3])/3.0
#print(M)

SM=np.zeros((len(M)-(Days-1),1))
for i in range((Days-1),len(M)):
    SM[i-(Days-1)]=np.average(M[i-(Days-1):i+1])
#print(SM)

D=np.zeros((len(M)-(Days-1),1))
for i in range((Days-1),len(M)):
    D[i-(Days-1)]=np.average(np.abs(M[i-(Days-1):i+1]-SM[i-(Days-1)]))
#print(D)

CCI=np.zeros((len(SM),1))
for i in range(len(SM)):
    CCI[i-(Days-1)]=(M[i+(Days-1)]-SM[i])/(0.015*D[i])
#print(CCI)  #10

result = np.zeros((len(StochasticD),1))

SMAA = SMAA[Days-1:,:]
result = SMAA

WMA = WMA[Days-1:,:]
result = np.append(result,WMA,axis=1)

Momentum = Momentum[Days-1:,:]
result = np.append(result,Momentum,axis=1)

StochasticK = StochasticK[Days-1:,:]
result = np.append(result,StochasticK,axis=1)

result = np.append(result,StochasticD,axis=1)

RSI = RSI[Days-2:,:]  #one extra cut because of Up and down
result = np.append(result,RSI,axis=1)

MACD = MACD[2*(Days-1):,:]  #one extra cut because of Up and down
result = np.append(result,MACD,axis=1)

LWR = LWR[2*(Days-1):,:]
result = np.append(result,LWR,axis=1)

ADO = ADO[2*Days-3:,:]
result = np.append(result,ADO,axis=1)

CCI =CCI[Days-1:,:]
result = np.append(result,ADO,axis=1)

y = y.reshape((len(X),1))
y = y[2*(Days-1):,:]
result = np.append(result,y,axis=1)

new_data = np.zeros((len(result)-1,12))

for i in range (0,len(result)-1) :
    for j in range(0,len(result[0])) :
        
        if j == 10 :
            if result[i][j] > result[i+1][j] :
                new_data[i][j] = 1
                new_data[i][j+1] = 0
            else :
                new_data[i][j] = 0
                new_data[i][j+1] = 1
        
        else :
            if result[i][j] > result[i+1][j] :
                new_data[i][j] = 0
            else :
                new_data[i][j] = 1
            
new_datadf = pd.DataFrame(new_data)
new_datadf.to_csv("train.csv",header=True)
