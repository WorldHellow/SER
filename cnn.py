from __future__ import print_function
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wav
from scipy.io import wavfile
import csv
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import more_itertools as mit
from numpy.fft import rfft
from matplotlib.mlab import find
from numpy import argmax, mean, diff, log
import os
import pickle


def loadModel(modelJson, modelH5):
	# load json and create model
	json_file = open(modelJson, 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights(modelH5)
	print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	score = loaded_model.evaluate(X, Y, verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


def cnnClassifier(source, modelSaveLocation):
	with open(source, 'rb') as f:
		dataset = pickle.load(f)

	batch_size = 32
	num_classes = 6
	epochs = 32
	num_predictions = 20
	model_name = 'emoModelCNN.h5'

	totalDataSamples = dataset["data"].shape[0]

	labelKeys = {'N':0, 'H':1, 'S':2, 'A':3, 'F':4, 'D':5}

	# The data, split between train and test sets:
	x_train, y_train = dataset["data"][:-int(1*(totalDataSamples*0.2))], dataset["targets"][:int(-1*(totalDataSamples*0.2))]
	x_test, y_test = dataset["data"][int(-1*(totalDataSamples*0.2)):], dataset["targets"][int(-1*(totalDataSamples*0.2)):]

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
					input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate SGD optimizer
	SGD = keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=1e-6)	

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
					optimizer=SGD,
					metrics=['accuracy'])

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	
	print('Not using data augmentation.')
	model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(x_test, y_test),
			shuffle=True)

	# serialize model to JSON
	model_json = model.to_json()
	with open(modelSaveLocation + model_name[:-3] + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(modelSaveLocation+model_name)
	print("Saved model to disk")


def wavToSpectogram(filePath, wavName, emotion):
	sampleRate, soundData = wav.load(filePath)	
	frequencies, times, spectrogram = signal.spectrogram(soundData, sampleRate)
	
	# plt.pcolormesh(times, frequencies, spectrogram)
	# plt.imshow(spectrogram)
	# plt.ylabel('Frequency [Hz]')
	# plt.xlabel('Time [sec]')
	# plt.show()
	
	cmap = plt.cm.jet
	
	plt.imsave(wavName+'_'+str(topIndex[index])+'_'+emotion+'.png', spectrogram, cmap=cmap)


def stftConversion(source, destination):
	fileHeaderAt = 1
	firstLetterOfEmo = 9
	# goodFiles = 7441 - 3

	ifile = open('/home/nauman/Documents/CREMA-D/processedResults/summaryTable.csv', "rt", encoding='us-ascii')
	reader = csv.reader(ifile)

	VECTOR = np.array([[]])
	LABELS = np.array([])

	for row in reader:
		if (row[0] != '' and row[0] != '6174' and row[0] != '3816' and row[0] != '5980'):
			# if (row[0] != '' and row[0] != '6174'):
			print("Working on file no :", row[0])
			fileSplit = (source+row[fileHeaderAt]+'.wav').split('/')
			wavName = fileSplit[-1][:-4] + ".wav"
			emotion = row[fileHeaderAt][firstLetterOfEmo]
			wavToSpectogram(source+wavName, destination+wavName, emotion)
	
	ifile.close()


def generateDataset(source, destination):
	fileHeaderAt = 1
	firstLetterOfEmo = -5

	IMAGES = []
	LABELS = np.array([])
	NAMES = np.array([])

	wavFiles = [f for f in os.listdir(source) if f.endswith('.png')]

	counter = 0
	for i in wavFiles:
		print("Working on file no :", counter)
		
		labelKeys = {'N':0, 'H':1, 'S':2, 'A':3, 'F':4, 'D':5}

		LABELS = np.append(LABELS, labelKeys[i[firstLetterOfEmo]])
		NAMES = np.append(NAMES, i[:-4])
		img = load_img(source+i)
		img = img_to_array(img)
		IMAGES.append(img)

		counter+=1
			
	LABELS = np.array(LABELS)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(LABELS)
	IMAGES = np.array(IMAGES)
	
	# print('IMAGES', IMAGES.shape)
	# print('LABELS', LABELS.shape)
	# print('NAMES', NAMES.shape)

	dataset = {"data":IMAGES, "targets":LABELS, "fileNames":NAMES}

	# pickle.dump(dataset, open(destination + 'creamaDataset' + '.pkl', "wb"))

	# with open(destination + 'creamaDataset' + '50' + '.pkl', 'wb') as f:
	# 	pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)