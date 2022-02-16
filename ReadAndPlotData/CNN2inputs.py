import numpy as np
import sys
#from pycbc import types
#from gwpy.timeseries import TimeSeries
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pydot
import pydotplus
from pydotplus import graphviz
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import imageio
def rgb2gray(rgb):
    return np.dot(rgb[...,:2], [0.2989, 0.5870])
#from tensorflow.keras.layers.convolutional import Conv2D
#from tensorflow.keras.layers.pooling import MaxPooling2D
#from tensorflow.keras.layers.merge import concatenate
datasetEntrenamiento = []
datasetPrueba = []
with open("./DatasetEntrenamiento.bin", "rb") as datasetfile:
    datasetEntrenamiento = pickle.load(datasetfile)
with open("./DatasetPrueba.bin", "rb") as datasetfile:
    datasetPrueba = pickle.load(datasetfile)
train_imagesL1 = []
train_labelsL1 = []
test_imagesL1 = []
test_labelsL1 = []
train_imagesH1 = []
train_labelsH1 = []
test_imagesH1 = []
test_labelsH1 = []
for i in range(0,len(datasetEntrenamiento)):
    for j in range(0,len(datasetEntrenamiento[i][0][0][0][1])):
        if(np.shape(datasetEntrenamiento[i][0][0][0][1][j]) == (28,50)):
            train_imagesL1.append(datasetEntrenamiento[i][0][0][0][1][j])
            train_labelsL1.append(datasetEntrenamiento[i][0][1][0][j])
            train_imagesH1.append(datasetEntrenamiento[i][0][0][0][0][j])
            train_labelsH1.append(datasetEntrenamiento[i][0][1][0][j])
        if(np.shape(datasetPrueba[0][0][0][0][1][j]) == (28,50)):
            test_imagesL1.append(datasetPrueba[0][0][0][0][1][j])
            test_labelsL1.append(datasetPrueba[0][0][1][0][j])
            test_imagesH1.append(datasetPrueba[0][0][0][0][0][j])
            test_labelsH1.append(datasetPrueba[0][0][1][0][j])

train_imagesL1 = np.array(train_imagesL1)
#train_imagesL1 = train_imagesL1.reshape(-1,28,50,1)
train_labelsL1 = np.array(train_labelsL1)
test_imagesL1 = np.array(test_imagesL1)
#test_imagesL1 = test_imagesL1.reshape(-1,28,50,1)
test_labelsL1 = np.array(test_labelsL1)
train_imagesH1 = np.array(train_imagesH1)
#train_imagesH1 = train_imagesH1.reshape(-1,28,50,1)
train_labelsH1 = np.array(train_labelsH1)
test_imagesH1 = np.array(test_imagesH1)
#test_imagesH1 = test_imagesH1.reshape(-1,28,50,1)
test_labelsH1 = np.array(test_labelsH1)
class_names = ['Noise','OG+Noise']
prueba = np.ndarray((len(train_imagesH1),28,50,2))
#print(prueba)
#[28][50]
train_imagesTotal2 = np.stack((train_imagesH1,train_imagesL1),axis=3)
train_imagesTotal2 = np.stack((test_imagesH1,test_imagesL1),axis=3)
train_imagesTotal = np.ndarray((len(train_imagesH1),28,50,2))
test_imagesTotal = np.ndarray((len(test_imagesH1),28,50,2))
for k in range(len(train_imagesH1)):
    for i in range(len(train_imagesH1[k])):
        for j in range(len(train_imagesH1[k][i])):
            train_imagesTotal[k][i][j][0] = train_imagesH1[k][i][j]
            train_imagesTotal[k][i][j][1] = train_imagesL1[k][i][j]
            test_imagesTotal[k][i][j][0] = test_imagesH1[k][i][j]
            test_imagesTotal[k][i][j][1] = test_imagesL1[k][i][j]
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28,50,2)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
plot_model(model, to_file='arquitectura.png',show_shapes=True)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_imagesTotal, train_labelsL1,batch_size=28,epochs=7)
test_loss, test_acc = model.evaluate(test_imagesTotal,test_labelsL1)
print("Acurracy",float(test_acc))
predictions = model.predict(test_imagesTotal)
predicted_label = np.zeros(len(predictions))

for i in range(len(predictions)):
    predicted_label[i] = int(np.argmax(predictions[i]))
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img = test_imagesTotal[i]     
    gray = rgb2gray(img)    
    plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    predicted = int(predicted_label[i])
    true_label = test_labelsL1[i]
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_names[predicted], class_names[true_label]), color = color)
plt.show()
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img = test_imagesTotal[i]     
    gray = rgb2gray(img)    
    plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    predicted = int(predicted_label[i])
    true_label = test_labelsL1[i]
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_names[predicted], class_names[true_label]), color = color)
plt.show()
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img = test_imagesTotal[i+25]     
    gray = rgb2gray(img)    
    plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    predicted = int(predicted_label[i+25])
    true_label = test_labelsL1[i+25]
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_names[predicted], class_names[true_label]), color = color)
plt.show()
