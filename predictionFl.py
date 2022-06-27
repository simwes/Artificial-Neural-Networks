from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.models import  model_from_json
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
import os, csv
import random, math

# ==============================
# Input:  relativeData.csv
# Output: Flx, Fly

random.seed(1)
# load the dataset
#path = 'dataTrain.csv'
path = 'assembleSelectedData/dataTest_fileNumber14.csv'
data = read_csv(path)

# Normalize data
# ensure all data is float
data = data.astype('float32')
scalerKine = MinMaxScaler( feature_range=(0, 1) )
dataTrain = scalerKine.fit_transform(data)

# split into input and output columns
frame = np.asarray(data)[:,0]
X_test = dataTrain[:,1:10]
y_test = dataTrain[:,-2:]

print('X')
print( np.asarray(data)[1,-2:] )
print('')

# split into train and test datasets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
print(  X_test.shape,  y_test.shape)


# define model

nLayers, nNeurons, RMSE = [], [], []

for iLayers in range(8, 9):
	for iNeurons in range(8, 9):

		# ============== LOAD MODEL - serialize model to JSON ===============
		# load json and create model
		json_file = open( "models/model_nLayers%d_nNeurons%d.json" %  (  iLayers, iNeurons ), 'r' )
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights( "models/model_nLayers%d_nNeurons%d.h5" %  (  iLayers, iNeurons ) )
		# compile the model
		model.compile( optimizer='adam', loss='mse' )

		print('')
		print( "Loaded model_nLayers%d_nNeurons%d.json" %  ( iLayers, iNeurons ) )

		FlxMean = np.mean(y_test[:,0])
		FlyMean = np.mean(y_test[:,1])
		FlMean = math.sqrt(FlxMean**2 + FlyMean**2 )
		print('FlMean: ', FlMean, 'FlxMean: ', FlxMean, 'FlyMean: ', FlyMean)
		error = model.evaluate(X_test, y_test, verbose=0)
		print( 'MSE: %.10f, RMSE: %.10f' % (error, sqrt(error)) )

		nLayers.append(iLayers)
		nNeurons.append(iNeurons)
		RMSE.append(sqrt(error))

# ------------------------------------------------------------------------------
# ==============================================================================

#-------------- write only for all the NN NOT for a single case ----------------
# with open('rmse.csv' , 'w') as f:
#    writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
#    writer.writerow( [     'nLayers', 'nNeurons', 'RMSE'  ] )
#    writer.writerows( zip(  nLayers,   nNeurons,   RMSE ) )

# make a prediction

FlxPred, FlyPred = [], []
FlxRef, FlyRef = [], []

for index in range( X_test[:,0].size ):

	row = X_test[index,:].tolist()

	yhat = model.predict([row])

	yhat = np.asarray( yhat )

	yhat = yhat.reshape( (yhat.shape[1]) )


	print( '')
	print( 'Predicted Flx:', yhat[0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2]  )
	FlxPred.append(yhat[0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2])
	print( 'Actual:', y_test[index,0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2] )
	FlxRef.append(y_test[index,0]/scalerKine.scale_[-2]  + scalerKine.data_min_[-2])

	print( '')
	print( 'Predicted Fly:', yhat[1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1]  )
	FlyPred.append(yhat[1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1])
	print( 'Actual:', y_test[index,1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1] )
	FlyRef.append(y_test[index,1]/scalerKine.scale_[-1]  + scalerKine.data_min_[-1])


with open('FlPrediction.csv' , 'w') as f:
   writer = csv.writer( f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
   writer.writerow( [       'frame','FLxPred', 'FLyPred', 'FLxRef', 'FLyRef'  ] )
   writer.writerows( zip(    frame,  FlxPred,   FlyPred,   FlxRef,   FlyRef ) )