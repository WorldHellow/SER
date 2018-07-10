from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import csv
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os
from rnn import *

np.set_printoptions(threshold=np.nan)


def main():
	source = "/home/nauman/Documents/CREMA-D/AudioWAV/"
	destination = "/home/nauman/Documents/preprocessedSpectogramData/"
	saveeModels = "/home/nauman/Dropbox/PROJECTS/audioSentimentAnalysis/saveeDemo/modelsDir/saveeModels/"
	featuresAndLabels = "/home/nauman/Documents/preparedData/test12/"
	saveeExtracted = "/home/nauman/Documents/preparedData/saveeExtracted/"
	saveeFeatures = "/home/nauman/Documents/preparedData/saveeFeatures/"
	saveeData = "/home/nauman/Documents/SAVEE/AudioData/"
	SAMPLE = "/home/nauman/Documents/CREMA-D/AudioWAV/" + "1028_IEO_SAD_LO.wav"
	datasetLocation = "/home/nauman/Documents/"
	listOfFiles = ["DC", "JE", "JK", "KL"]

	# stftConversion(source, destination)
	# generateDataset(destination, "data/")
	rnnClassifier("data/"+'creamaDataset.pkl', "models/")


main()
