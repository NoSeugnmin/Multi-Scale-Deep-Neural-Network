# _*_ coding: utf-8 _*_
# All Includes

import numpy
import codecs
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, LSTM, Bidirectional, SimpleRNN, GRU, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, LSTM, MaxPooling1D
import numpy as np
numpy.set_printoptions(threshold=numpy.nan)

seed = 7
dataframe = pandas.read_csv('sum_modify_magnetic.csv',engine='python', delimiter=',' ,header=0)
# dataframe = pandas.read_csv('only_wifi.csv',engine='python', delimiter=',' ,header=0)
# dataframe = pandas.read_csv('only_magnetic.csv',engine='python', delimiter=',' ,header=0)

#plt.plot(train_data)
#plt.show()
look_back=1
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    window_slice = 1
    timesteps = 1
    for i in range(window_slice, len(dataset), timesteps):
        a = dataset[i-window_slice:(i+look_back), 0:46]
        b = dataset[i:(i+look_back),46]
        dataX.append(a)
        dataY.append(b)
        #print("data X : ", dataX)

    return numpy.array(dataX), numpy.array(dataY)

#print (dataX)
#print (dataY)



dataset = dataframe.values
dataset = dataset.astype('float32')
#print("dataset : ", dataset)
#scaler = MinMaxScaler(feature_range=(0, 11))
#dataset = scaler.fit_transform(dataset)


train_size = int(len(dataset) * 0.78)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
#print("trainX Start : ", trainX)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

#print (trainX)
#print(trainX)
#print (trainX.shape) #98, 1, 16

#print (testX)
#print (testX.shape) #25, 1, 16


trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
#print("train X : numpy", trainX)
#print("test X : numpy", testX)


#print ("test Y tupe : ", testY.shape)
#print ("train Y type : ", trainY.shape) #98, 6

# LSTM Neural Network's internal structure
n_steps, n_length = 4, 32
n_features = 3
batch_size = 218
epochs = 300
from keras.layers import TimeDistributed
from keras.layers import Dropout, SimpleRNN
from keras.layers import Flatten
model = Sequential()
model.add(LSTM(256,input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
# model.add(SimpleRNN(256,input_shape=(trainX.shape[1], trainX.shape[2])))
#print("trainX shape ", trainX.shape[1], trainX.shape[2])
model.add(LSTM(128))
# model.add(Dense(128))
# model.add(SimpleRNN(128, kernel_initializer='uniform',activation= 'relu'))
model.add(Dense(34, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(testX, testY), shuffle=False)

# summarize performance of the model
scores1 = model.evaluate(trainX, trainY, verbose=0)
print("Train Accuracy: %.2f%%" % (scores1[1] * 100))

scores2 = model.evaluate(testX, testY, verbose=0)
print("Test Accuracy: %.2f%%" % (scores2[1] * 100))

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# make predictions label
confusion_trainPredict = model.predict_classes(trainX)
confusion_testPredict = model.predict_classes(testX)
# Classification Report & Confusion Matrix

target_names = ['class 0(RSSI_1)', 'class 1(RSSI_2)', 'class 2(RSSI_3)', 'class 3(RSSI_4)',
                'class 4(RSSI_5)', 'class 5(RSSI_6)', 'class 6(RSSI_7)', 'class 7(RSSI_8)',
                'class 8(RSSI_9)', 'class 9(RSSI_10)', 'class 10(RSSI_11)', 'class 11(RSSI_12)',
                'class 12(RSSI_13)', 'class 13(RSSI_14)', 'class 14(RSSI_15)', 'class 15(RSSI_16)',
                'class 16(RSSI_17)', 'class 17(RSSI_18)', 'class 18(RSSI_19)','class 19(RSSI_20)', 'class 20(RSSI_21)', 'class 21(RSSI_22)', 'class 22(RSSI_23)',
                'class 23(RSSI_24)', 'class 24(RSSI_25)', 'class 25(RSSI_26)', 'class 26(RSSI_27)',
                'class 27(RSSI_28)', 'class 28(RSSI_29)', 'class 29(RSSI_30)','class 30(RSSI_31)', 'class 31(RSSI_32)',
                'class 32(RSSI_33)', 'class 33(RSSI_34)']

#print ("Classification Report(test):")
#print(classification_report(numpy.argmax(testY, axis=1), confusion_testPredict, target_names=target_names))

print ("Confusion Matrix(test):")
print (confusion_matrix(numpy.argmax(testY, axis=1), confusion_testPredict))



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()

from sklearn import metrics
import seaborn as sns
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=target_names,
                yticklabels=target_names,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


y_pred_test = model.predict(testX)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(testY, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))
scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy : ", scores[1]*100)
