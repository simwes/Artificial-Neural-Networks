from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import numpy  as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
import datetime
import pickle 
import pandas as pd
import os
import random, math

# ==============================
# Input:  relativeData.csv
# Output: Flx, Fly

random.seed(1)
# load the dataset
path = 'dataTrain.csv'
data = read_csv(path)

# Normalize data
# ensure all data is float
data = data.astype('float32')
scalerKine = MinMaxScaler( feature_range=(0, 1) )
dataTrain = scalerKine.fit_transform(data)

# split into input and output columns
X = dataTrain[:,1:10]
y = dataTrain[:,-2:]

print('X')
print( np.asarray(data)[1,-2:] )
print('')

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]

# define model

for iLayers in range(14, 15):
	for iNeurons in range(7,8):

		model = Sequential()
		model.add( Dense(n_features, activation = 'relu', kernel_initializer = 'he_normal', input_shape=(n_features,)) )
		for i in range(iLayers):
			print('Add Layer', i+1)
			model.add( Dense(iNeurons, activation = 'relu', kernel_initializer = 'he_normal') )

		model.add( Dense(iNeurons, activation = 'relu', kernel_initializer = 'he_normal') )
		model.add( Dense(2) )

		# compile the model
		model.compile( optimizer='adam', loss='mse' )

		histFile =  "logHist.csv"
		csv_logger = CSVLogger(histFile, append=True, separator=',')

		history = model.fit( X_train, 
				  	 		 y_train, 
				   	 		 epochs = 500, 
				   			 batch_size = 32,
				   			 validation_split=0.1, 
				   			 verbose = 1,
				   			 callbacks=[csv_logger] )


		# ============== SAVE MODEL - serialize model to JSON ===============
		# ------------------------------------------------------------------
		model_json = model.to_json()
		with open("models/model_nLayers%d_nNeurons%d.json" %  (  iLayers, iNeurons ), "w") as json_file:
		   		 json_file.write(model_json)

		# serialize weights to HDF5
		model.save_weights( "models/model_nLayers%d_nNeurons%d.h5" %  (  iLayers, iNeurons ) )

		# Save loss hystory
		hist_df = pd.DataFrame(history.history)
		with open('models/hystory_nLayers%d_nNeurons%d.csv' %  (  iLayers, iNeurons ), 'w') as file_pi:
			hist_df.to_csv(file_pi)

		print( "Saved model_nLayers%d_nNeurons%d" %  (  iLayers, iNeurons ) )

		# evaluate the model

		error = model.evaluate(X_test, y_test, verbose=0)
		print( 'MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)) )

		FlxMean = np.mean(y_test[:,0])
		FlyMean = np.mean(y_test[:,1])
		FlMean = math.sqrt(FlxMean**2 + FlyMean**2 )
		print('FlMean: ', FlMean, 'FlxMean: ', FlxMean, 'FlyMean: ', FlyMean)
		error = model.evaluate(X_test, y_test, verbose=0)
		print( 'MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)) )
# ------------------------------------------------------------------
# ==================================================================

# make a prediction

# index = 10
# row = X_test[index,:].tolist()

# yhat = model.predict([row])

# # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

# yhat = np.asarray( yhat )

# yhat = yhat.reshape( (yhat.shape[1]) )

# print( 'yhat.shape',yhat.shape, 'X_test[index,:].shape', X_test[index,:].shape )

# #inv_yhat = concatenate((yhat, X_test[0, :]))

# #Xtract the coefiecent from from MinMaxScaler to scale back

# # print('')
# # print( 'Predicted 1:', yhat[0]/scaler.scale_[-2]  + scaler.data_min_[-2] )
# # print('Actual:', y_test[0]/scaler.scale_[-2]  + scaler.data_min_[-2])

# # print( 'Predicted 2:', yhat[1]/scaler.scale_[-1]  + scaler.data_min_[-1] )
# # print('Actual:', y_test[1]/scaler.scale_[-1]  + scaler.data_min_[-1])

# print(y_test.shape)
# print( '')
# print( 'Predicted Flx:', yhat[0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2]  )
# print( 'Actual:', y_test[index,0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2] )
# print( '')
# print( 'Predicted Fly:', yhat[1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1]  )
# print( 'Actual:', y_test[index,1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1] )
