# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 09:13:33 2016

@author: anand
"""

import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('nik225.csv')
df['Date'] = pd.to_datetime(df['Date'])
examples = 8059
df = df.iloc[0:examples,:]
ts = df['Close']
dDays = 10

# original plotting of the data
plt.title('Closing Value vs Date')
plt.xlabel('Year')
plt.ylabel('Closing Value')
plt.plot(df['Date'],df['Close'])

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=dDays)
    rolstd = pd.rolling_std(timeseries, window=dDays)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
test_stationarity(ts)

#log transformation 
ts_log = numpy.log(ts)

#moving avg
moving_avg = pd.rolling_mean(ts_log,dDays)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

#weighted avg
expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

#1st order differencing
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

#can try 2nd and 3rd order differencing in similar fashion
#also take note of test-statistics with critical value


#decomposition
df.reset_index(inplace=True)
df = df.set_index('Date')
decomposition = seasonal_decompose(numpy.log(df.Close),freq=12)  
#decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

'''plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()'''

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

ts.to_csv("ts.csv",header=True)
ts_log.to_csv("ts_log.csv",header=True)
ts_log_moving_avg_diff.to_csv("ts_log_moving_avg_diff.csv",header=True)
ts_log_ewma_diff.to_csv("ts_log_ewma_diff.csv",header=True)
ts_log_decompose.to_csv("ts_log_decompose.csv",header=True)



