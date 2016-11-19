library(forecast)
library(tseries)
library(urca)
library(ggplot2)

RMSFunction <- function(original,predicted){
  diff = original - predicted
  diff = diff ^2
  result = sqrt(mean(diff))
  return(result)
}

df=read.csv('nik225.csv')

days = 10

#partition each data such that last d-days are saved in testing and left all in training.

#Original Data
ts = read.csv('ts.csv')   #0 1 2
train_length = length(ts$Close) - days
total_length = length(ts$Close)
ts_train = ts$Close[1:train_length]
ts_test = ts$Close[(train_length+1):total_length]

#logarithmic Data
ts_log = read.csv('ts_log.csv')
train_length = length(ts_log$Close) - days
total_length = length(ts_log$Close)
ts_log_train = ts_log$Close[1:train_length]
ts_log_test = ts$Close[(train_length+1):total_length]

#moving Average Data
ts_log_moving_avg_diff = read.csv('ts_log_moving_avg_diff.csv')
train_length = length(ts_log_moving_avg_diff$Close) - days
total_length = length(ts_log_moving_avg_diff$Close)
ts_log_moving_avg_diff_train = ts_log_moving_avg_diff$Close[1:train_length]
ts_log_moving_avg_diff_test = ts_log_moving_avg_diff$Close[(train_length+1):total_length]


# exponential moving average data
ts_log_ewma_diff = read.csv('ts_log_ewma_diff.csv')
train_length = length(ts_log_ewma_diff$Close) - days
total_length = length(ts_log_ewma_diff$Close)
ts_log_ewma_diff_train = ts_log_ewma_diff$Close[1:train_length]
ts_log_ewma_diff_test = ts_log_ewma_diff$Close[(train_length+1):total_length]


# decomposed data
ts_log_decompose = read.csv('ts_log_decompose.csv')
train_length = length(ts_log_decompose$Close) - days
total_length = length(ts_log_decompose$Close)
ts_log_decompose_train = ts_log_decompose$Close[1:train_length]
ts_log_decompose_test = ts_log_decompose$Close[(train_length+1):total_length]


#adf test on each of the above data 
adf_ts = adf.test(ts_train)
adf_ts_log = adf.test(ts_log_train)
adf_ts_log_moving_avg_diff = adf.test(ts_log_moving_avg_diff_train)
adf_ts_log_wema_diff = adf.test(ts_log_ewma_diff_train)
adf_ts_log_decompose = adf.test(ts_log_decompose_train)

# p-values for all the different datasets
# cat(adf_ts$p.value,"\n")
# cat(adf_ts_log$p.value,"\n")
# cat(adf_ts_log_moving_avg_diff$p.value,"\n")
# cat(adf_ts_log_wema_diff$p.value,"\n")
# cat(adf_ts_log_diff$p.value,"\n")
# cat(adf_ts_log_decompose$p.value,"\n")


# perform training & testing for each data now
data <- list()
data[[1]] <- ts_train  #0 1 2
data[[2]] <- ts_log_train # 0 1 0
data[[3]] <- ts_log_moving_avg_diff_train # 2 0 3
data[[4]] <- ts_log_ewma_diff_train  #1 0 0
data[[5]] <- ts_log_decompose_train  # 2 0 1


index = 1
format = ".png"

for (currentdata in data){ 
  TS = currentdata
  #TS = ts$Close
  
  
  final.aic <- Inf
  final.bic <- Inf
  final.order <- c(0,0,0)
  for (i in 0:3){
      for (j in 0:4){
        
        if(inherits(arima(TS,order=c(i,0,j)),"try-error")){
          break
        }
        model = arima(TS,order=c(i,0,j))  #for ARMA, I=0
        current.aic <- AIC(model)
        current.bic <- BIC(model)
        
        if(current.bic < final.bic && current.aic<final.aic){
          final.aic <-current.aic
          final.bic <-current.bic
          final.order <- c(i,0,j)
          final.accuracy <- accuracy(model)
          if(inherits(arima(TS,order=c(i,0,j)),"try-error")){
            break
          }
          else{
            final.arima <- arima(TS,order=c(i,0,j))
          }
        }
      }
    
  }
  cat("Optimum Parameters are:",final.order,"  MASE Error = ",final.accuracy[1,"MASE"],"\n")
  #cat("Loop over\n")
  #cat(final.order,final.bic,final.aic,"\n")
  
  ts_forecast = forecast(final.arima)
  ts_original = ts_forecast$x
  ts_predicted = ts_forecast$fitted
  

  # png(filename = paste(index,"PREDICT",format,sep='_'))
  # fit = arima(TS,final.order)
  # plot(TS,type='l',xlim=c(7590,8050),col='black')
  # pr1<-predict(fit,n.ahead=days)
  # lines(pr1$pred,col="red")
  # lines(pr1$pred+2*pr1$se,col="red",lty=3)
  # lines(pr1$pred-2*pr1$se,col="red",lty=3)
  # par(new=FALSE)
  # dev.off()
  
  index = index + 1  
}

#Integrated ARMA I=0
#1 c(0,1,2) xlim=c(6900,8040)
#2 c(1,0,0) xlim=c(6900,8040)
#3 c(4,0,2) xlim=c(7600,8040)
#4 c(1,0,0) xlim=c(7600,8040)
#6 c(3,0,4) xlim=c(7700,8040)

# fit = arima(TS,final.order)
# plot(TS,type='l',xlim=c(7600,8040),col='black')
# pr1<-predict(fit,n.ahead=10)
# lines(pr1$pred,col="red")
# lines(pr1$pred+2*pr1$se,col="red",lty=3)
# lines(pr1$pred-2*pr1$se,col="red",lty=3)
