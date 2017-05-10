The model is done as follows.

1) ARIMA model

	i) First run preprocess.py with command  : python preprocess.py
		This will create 5 files which correspond to different data. Original,Log,Moving Averge,Exponential Average & Decomposed
		For the convenience we have already included all the files already in the directory.

	ii) Now run once - arma.R and arima.R one by one. which will calculate the optimum (p,i,q) value in case of arima and (p,0,q) in case 			of arma.R. These files will also predict the share price on the next 10-days. Again for the convenience, all the codes which 			produce and save graphs have been either commented or made manually.

2) ANN 

	i) First run Preprocessing_of_data.py with the command : python Preprocessing_of_data.py This file will generate all the technical 			parameters like RollingAvg, MovingAvg etc. in the file train.csv
	ii) ANN.py now will take train.csv as a input and train the neural network on parameters to produce the accuracy.

3) SVC

	i) First run Preprocessing_of_data.py with the command : python Preprocessing_of_data.py This file will generate all the technical 			parameters like RollingAvg, MovingAvg etc. in the file train.csv
	ii) SVC.py will now take train.csv as a input and train the Support vector classifier to produce accuracy on given data.
