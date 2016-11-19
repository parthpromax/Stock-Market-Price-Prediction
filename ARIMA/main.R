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

ts = read.csv('ts.csv')   #0 1 2
train_length = length(ts$Close) - days
total_length = length(ts$Close)
ts_train = ts$Close[1:train_length]
ts_test = ts$Close[(train_length+1):total_length]

ts_log = read.csv('ts_log.csv')
train_length = length(ts_log$Close) - days
total_length = length(ts_log$Close)
ts_log_train = ts_log$Close[1:train_length]
ts_log_test = ts$Close[(train_length+1):total_length]

ts_log_moving_avg_diff = read.csv('ts_log_moving_avg_diff.csv')
train_length = length(ts_log_moving_avg_diff$Close) - days
total_length = length(ts_log_moving_avg_diff$Close)
ts_log_moving_avg_diff_train = ts_log_moving_avg_diff$Close[1:train_length]
ts_log_moving_avg_diff_test = ts_log_moving_avg_diff$Close[(train_length+1):total_length]


ts_log_ewma_diff = read.csv('ts_log_ewma_diff.csv')
train_length = length(ts_log_ewma_diff$Close) - days
total_length = length(ts_log_ewma_diff$Close)
ts_log_ewma_diff_train = ts_log_ewma_diff$Close[1:train_length]
ts_log_ewma_diff_test = ts_log_ewma_diff$Close[(train_length+1):total_length]

ts_log_decompose = read.csv('ts_log_decompose.csv')
train_length = length(ts_log_decompose$Close) - days
total_length = length(ts_log_decompose$Close)
ts_log_decompose_train = ts_log_decompose$Close[1:train_length]
ts_log_decompose_test = ts_log_decompose$Close[(train_length+1):total_length]


adf_ts = adf.test(ts_train)
adf_ts_log = adf.test(ts_log_train)
adf_ts_log_moving_avg_diff = adf.test(ts_log_moving_avg_diff_train)
adf_ts_log_wema_diff = adf.test(ts_log_ewma_diff_train)
adf_ts_log_decompose = adf.test(ts_log_decompose_train)

# cat(adf_ts$p.value,"\n")
# cat(adf_ts_log$p.value,"\n")
# cat(adf_ts_log_moving_avg_diff$p.value,"\n")
# cat(adf_ts_log_wema_diff$p.value,"\n")
# cat(adf_ts_log_decompose$p.value,"\n")


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
  #cat(index,"\n")
  
  final.aic <- Inf
  final.bic <- Inf
  final.order <- c(0,0,0)
  for (i in 0:3){
    for (d in 1:3){ #dmax = 3 only 3 times differenced the data
      for (j in 0:4){
        #cat(i,d,j,"\n")
        if(inherits(arima(TS,order=c(i,d,j)),"try-error")){
          break
        }
        model = arima(TS,order=c(i,d,j))
        current.aic <- AIC(model)
        current.bic <- BIC(model)
        
        if(current.bic < final.bic && current.aic<final.aic){
          #cat(i,d,j,current.aic,current.bic,"\n")
          final.aic <-current.aic
          final.bic <-current.bic
          final.order <- c(i,d,j)
          if(inherits(arima(TS,order=c(i,d,j)),"try-error")){
            break
          }
          else{
            final.arima <- arima(TS,order=c(i,d,j))
            #final.model <- final.arima
          }
        }
      }
    }
  }
  #cat("Loop over\n")
  #cat(final.order,final.bic,final.aic,"\n")
  
  ts_forecast = forecast(final.arima)
  ts_original = ts_forecast$x
  ts_predicted = ts_forecast$fitted
  

  # uncomment this code to generate plots of prediction
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






