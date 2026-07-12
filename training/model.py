import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import seaborn as sns

from setuptools import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, LSTM, Input, Reshape
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


def save(objectToStore, fileName):
	to = open("/{0}".format(fileName), "wb")
	pickle.dump(objectToStore, to)
	to.close()


def load(fileName):
	fileLocation = open("/{0}".format(fileName), "rb")
	object = pickle.load(fileLocation)
	fileLocation.close()
	return object


def isAcceptableFile(filePath):
	return os.path.isfile(filePath) and (".csv" in filePath) and (".icloud" not in filePath)


def windowCoordinates(allCoordinates, filePath):
	windowedCoordinates = []
	correspondingLabels = []

	for i in range(0, len(allCoordinates), windowSize):
		if i + windowSize < len(allCoordinates):
			windowedCoordinates.append(allCoordinates[i:i + windowSize])

			labelEnd = filePath.rfind("/")
			labelStart = filePath[:labelEnd].rfind("/")
			label = int(filePath[labelStart + 1: labelEnd])

			correspondingLabels.append(label)

	if len(windowedCoordinates) != len(correspondingLabels):
		raise Exception("The number of labels does not match the numner of training examples for: ", filePath)

	return windowedCoordinates, correspondingLabels


def getCoordinatesAndLabels(df, filePath):
	xyzCoor = df.iloc[:, :].values.astype(float)
	allCoordinates = [[row[0], row[1], row[2]] for row in xyzCoor if rowIsFreeOfNaNs(row)]
	return windowCoordinates(allCoordinates, filePath)


def rowIsFreeOfNaNs(row):
	return not math.isnan(row[0]) and not math.isnan(row[1]) and not math.isnan(row[2])
				 

def splitIntoSets(trainingSetDir):
	data = []
	labels = []
	directory = trainingSetDir
	for filePath in glob.iglob(directory, recursive=True):
		if isAcceptableFile(filePath):
			df = pd.read_csv(filePath)
			coordinates, correspondingLabels = getCoordinatesAndLabels(df, filePath)
			
			for i, _ in enumerate(coordinates): data.append(coordinates.pop(i))
			for i, _ in enumerate(correspondingLabels): labels.append(correspondingLabels.pop(i))


	print('\n\n\n Shape:', np.array(data).shape)
	print('Labels Shape:', np.array(labels).shape, '\n')

	X_train, X_test, Y_train, Y_test = train_test_split(np.array(data), np.array(labels), test_size=0.2, shuffle=True)
	return X_train, X_test, Y_train, Y_test


def trainModel(X_train, Y_train, X_test, Y_test):
	X_train_CNN = X_train.reshape(len(X_train), len(X_train[0]), len(X_train[0][0]), 1)
	X_val_CNN = X_val.reshape(len(X_val), len(X_val[0]), len(X_val[0][0]), 1)							 
	X_test_CNN = X_test.reshape(len(X_test), len(X_test[0]), len(X_test[0][0]), 1)

	cnn = tf.keras.Sequential([
			Input(shape=(windowSize,3,1), name='input'),
			Conv2D(50, (3, 3), activation='relu', padding='same'),
			MaxPooling2D((2, 2), padding='same'),
			Flatten(),
			Dense(12, activation='relu'),
			Dense(32, activation='relu'),
			Dense(numberOfActivities, activation='softmax')
		])  
	
	regular = tf.keras.Sequential([	 
			Flatten(),
			Dense(100, activation='relu'),
			Dense(100, activation='relu'),
			Dense(100, activation='relu'),
			Dense(32, activation='relu'),
			Dense(numberOfActivities, activation='softmax')
		])

	lstm = tf.keras.Sequential([
			Input(shape=(windowSize,3), name='input'),
			LSTM(50, activation='tanh', return_sequences=True),
			Flatten(),
			Dense(10, activation='relu'),
			Dense(32, activation='relu'),
			Dense(numberOfActivities, activation='softmax')
		])

	cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
	regular.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
	lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

	numberOfEpochs = epochs

	_h1 = cnn.fit(X_train_CNN, Y_train, epochs=numberOfEpochs, verbose=1, validation_data=(X_val_CNN, Y_val))
	_h2 = regular.fit(X_train, Y_train, epochs=numberOfEpochs, verbose=1, validation_data=(X_val, Y_val))
	_history = lstm.fit(X_train, Y_train, epochs=numberOfEpochs, verbose=1, validation_data=(X_val, Y_val))

	plot_model(cnn, to_file='CNN.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
	plot_model(regular, to_file='NN.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
	plot_model(lstm, to_file='LSTM.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

	# X_test = load("X_test")
	# Y_test = load("Y_test")

	print("\n\nTESTING: ")
	cnn.evaluate(X_test_CNN, Y_test, verbose=2)
	regular.evaluate(X_test, Y_test, verbose=2)
	lstm.evaluate(X_test, Y_test, verbose=2)

	cnn.summary()
	regular.summary()
	lstm.summary()

	getConfusionMatrix(X_test_CNN, Y_test, cnn)
	getConfusionMatrix(X_test, Y_test, regular)
	getConfusionMatrix(X_test, Y_test, lstm)
	return lstm


def getConfusionMatrix(data, labels, model):
	preds = model.predict(data)
	preds = tf.argmax(preds, axis=1).numpy()

	confusionMatrix = tf.math.confusion_matrix(labels=labels, predictions=preds, num_classes=numberOfActivities)
	confusionMatrix = np.array(confusionMatrix)
	normalizedConfusionMatrix = confusionMatrix/np.sum(confusionMatrix, axis=1, keepdims=True)

	classes = ['SJ', 'SFR', 'LR', 'SER', 'SIR', 'S']

	_, ax = plt.subplots(figsize=(8, 6))

	sns.heatmap(normalizedConfusionMatrix, annot=True, fmt='.3%', xticklabels=classes, yticklabels=classes, cmap='Blues')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('True')
	plt.show()


def convertToTfLite(X_train, Y_train, X_test, Y_test, modelName):
	model = trainModel(X_train, Y_train, X_test, Y_test)

	saveTFmodel(model, modelName)
	converter = tf.lite.TFLiteConverter.from_saved_model(modelName)

	converter.optimizations = [tf.lite.Optimize.DEFAULT]

	converter.representative_dataset = lambda:representative_dataset(X_train)

	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

	converter._experimental_lower_tensor_list_ops = False
	converter.experimental_new_converter = True
	converter.allow_custom_ops = True

	converter.target_spec.supported_types = [tf.int8]

	converter.inference_type = tf.int8
	converter.inference_input_type = tf.int8 
	converter.inference_output_type = tf.int8

	tfliteModel = converter.convert()

	checkScaleAndZeroPoint(tfliteModel)
	testAccurancyOnTfliteModel(model, tfliteModel, X_test)

	return tfliteModel


def representative_dataset(trainingSet):
	for value in trainingSet:
		value = value.reshape(1, len(trainingSet[0]), len(trainingSet[0][0]))
		yield [value.astype(np.float32)]


def saveTFmodel(model, modelName):
	run_model = tf.function(lambda x: model(x))
	BATCH_SIZE = 1
	STEPS = len(X_train[0])
	INPUT_SIZE = len(X_train[0][0])
	concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
	model.save(modelName, save_format="tf", signatures=concrete_func)


def testAccurancyOnTfliteModel(model, tfliteModel, X_test):
	# Initialise the interpreter that would be deployed on Arduino
	# using the tflite model
	interpreter = tf.lite.Interpreter(model_content=tfliteModel)
	interpreter.allocate_tensors()
	inputDetails = interpreter.get_input_details()
	outputDetails = interpreter.get_output_details()

	# Retrieve the scaling values that would be used on Arduino
	inputScale, inputZeroPoint = inputDetails[0]["quantization"]
	outputScale, outputZeroPoint = outputDetails[0]["quantization"]

	finalResult = []

	for testExample in X_test:

		# Predict an example with the regular float-based model
		testExample = testExample.reshape(1,len(X_test[0]),len(X_test[0][0]))
		expectedTensor = model.predict(testExample, verbose=0)
		expected = np.argmax(expectedTensor[0])

		# Predict an example with the quantized integer-based model
		testExample = (testExample / inputScale) + inputZeroPoint
		testExample = testExample.astype(np.int8).reshape(1,len(X_test[0]),len(X_test[0][0]))

		interpreter.set_tensor(inputDetails[0]["index"], testExample)
		interpreter.invoke()
		predictionTensor = interpreter.get_tensor(outputDetails[0]["index"])

		# Convert the result of the integer model back to a float value
		predictionTensor = (predictionTensor.astype(np.float32) - outputZeroPoint) * outputScale
		prediction = np.argmax(predictionTensor[0])

		# Compare the two predictions
		if expected == prediction:
			finalResult.append(1)
		else:
			finalResult.append(0)

		# Reset the interpreter so that a new example can be tested
		interpreter.reset_all_variables()

	finalAccuracy = (sum(finalResult)/len(finalResult))*100	
	print('\n\nQuantized model accuracy:', str(finalAccuracy)+'%\n\n')


def checkScaleAndZeroPoint(tfliteModel):
	interpreter = tf.lite.Interpreter(model_content=tfliteModel)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_scale, input_zero_point = input_details[0]["quantization"]
	output_scale, output_zero_point = output_details[0]["quantization"]

	print('\nScale:', input_scale, "\nZero point:", input_zero_point)
	print('\nOutput Scale:', output_scale, "\nOutput Zero point:", output_zero_point)


def hex_to_c_array(hex_data, var_name):
	c_str = ''
	c_str += '#ifndef ' + var_name.upper() + '_H\n'
	c_str += '#define ' + var_name.upper() + '_H\n\n'
	c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
	c_str += 'unsigned char ' + var_name + '[] = {'

	hex_array = []

	for i, val in enumerate(hex_data) :

		hex_str = format(val, '#04x')

		if (i + 1) < len(hex_data):
			hex_str += ','
		if (i + 1) % 12 == 0:
			hex_str += '\n '
		hex_array.append(hex_str)

	c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'
	c_str += '#endif //' + var_name.upper() + '_H'

	return c_str


def createCFile(X_train, Y_train, X_test, Y_test):
	c_model_name = "quantized_model"
	tflite_model = convertToTfLite(X_train, Y_train, X_test, Y_test, c_model_name)
	open(c_model_name+'.tflite', "wb").write(tflite_model)

	# headerFile = hex_to_c_array(tflite_model, c_model_name)
	# with open(c_model_name + '.h', 'w') as file:
	# 	file.write(headerFile)

	
datasetDirectory = '~/dataset/**'	
numberOfActivities = 6
epochs = 100
windowSize = 75

X_train, X_test, Y_train, Y_test = splitIntoSets(datasetDirectory)

middle = int(len(X_test)/2)

X_val = X_test[:middle]
Y_val = Y_test[:middle]

X_test = X_test[middle:]
Y_test = Y_test[middle:]

# save(X_test, "X_test")
# save(Y_test, "Y_test")

accelerometerRange = 4

X_train = X_train/accelerometerRange
X_test = X_test/accelerometerRange

createCFile(X_train, Y_train, X_test, Y_test)