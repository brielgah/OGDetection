import numpy as np
import sys
#from pycbc import types
#from gwpy.timeseries import TimeSeries
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
#import pandas as pd
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("matriz H1")

datasetEntrenamiento = []
datasetPrueba = []
with open("./DatasetEntrenamiento.bin", "rb") as datasetfile:
    datasetEntrenamiento = pickle.load(datasetfile)
with open("./DatasetPrueba.bin", "rb") as datasetfile:
    datasetPrueba = pickle.load(datasetfile)
train_images = []
train_labels = []
test_images = []
test_labels = []
for i in range(0,len(datasetEntrenamiento)):
    for j in range(0,len(datasetEntrenamiento[i][0][0][0][0])):
        if(np.shape(datasetEntrenamiento[i][0][0][0][0][j]) == (28,50)):
            train_images.append(datasetEntrenamiento[i][0][0][0][0][j])
            train_labels.append(datasetEntrenamiento[i][0][1][0][j])
        if(np.shape(datasetPrueba[0][0][0][0][0][j]) == (28,50)):
            test_images.append(datasetPrueba[0][0][0][0][0][j])
            test_labels.append(datasetPrueba[0][0][1][0][j])

train_images = np.array(train_images) / 255.0
train_labels = np.array(train_labels)
test_images = np.array(test_images) / 255.0
test_labels = np.array(test_labels)
plt.figure()
plt.imshow(test_images[i],cmap=plt.cm.binary)
plt.savefig("Imagen")
plt.close()
class_names = ['Noise','OG+Noise']
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28,50,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
train_images = train_images.reshape(-1,28,50,1)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#utils.plot_model(model, 'my_first_model.png', show_shapes=True)
history = model.fit(train_images, train_labels,batch_size=28,epochs=10) 
                    #validation_data=(test_images, test_labels))
test_images = test_images.reshape(-1,28,50,1)
test_loss, test_acc = model.evaluate(test_images,test_labels)
print("Acurracy",float(test_acc))
predictions = model.predict(test_images)
print(predictions.shape)
predicted_label = np.zeros(len(predictions))

for i in range(len(predictions)):
    predicted_label[i] = int(np.argmax(predictions[i]))

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    predicted = int(predicted_label[i])
    true_label = test_labels[i]
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_names[predicted], class_names[true_label]), color = color)
plt.savefig("H1")
plt.close()
c_matrix = metrics.confusion_matrix(test_labels,predicted_label)
cm_labels=['Ruido', 'Ruido + GW']   
plot_confusion_matrix(cm=c_matrix,classes=cm_labels,title='Matriz de confusion H1')
