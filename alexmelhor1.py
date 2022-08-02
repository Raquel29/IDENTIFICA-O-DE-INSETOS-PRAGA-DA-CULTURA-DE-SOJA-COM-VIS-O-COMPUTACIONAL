
from tensorflow import keras
import numpy as np
import os, sys
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,Sequential 
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import re
import glob, os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


#Pegando diretorio com as imagens 
from glob import glob
os.chdir("/home/raquel/ImagensFinais/ImagensFinais" )
img_aumenta = glob(os.path.join(os.getcwd(),'*.jpg'))

print("lista de imagens criadas com sucesso")
# rotulando como praga ou inseto
import cv2
data =[]
labels = []
Praga1 = 'Euschistus'
Praga2 = 'Nezara'
Praga3 = 'Pizedouros'
for i in img_aumenta:
  label = i.split(os.path.sep)[-1]
  if label.startswith(Praga1) or label.startswith(Praga2)  or label.startswith(Praga3) :
    label = label.replace(label, 'praga')
  else: 
    label = label.replace(label,'nãopraga')
 # print(label)
  image = cv2.imread(i)
  # update the data and labels lists, respectively
  data.append(image)
  labels.append(label)

print("Imagens rolutadas com sucesso")  

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

#data = tf.data.Dataset.from_tensor_slices(data)

print("Categorização feita com sucesso")  

#Dividindo em treino e teste
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,test_size=0.30, random_state=42)


print('Divisão de dados feita com sucesso')

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

print('Conversão para tensor feita com sucesso')

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy() 
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy() 
#validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy() 
print( "Tamanho dos dados de treinamento:", train_ds_size) 
print("Tamanho dos dados de teste:", test_ds_size) 
#print("Tamanho dos dados de validação:", validation_ds_size)

def processamento(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds = (train_ds
                  .map(processamento) 
                  .shuffle(buffer_size=train_ds_size) 
                  .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds 
                  .map(processamento) 
                  .shuffle(buffer_size=train_ds_size) 
                  .batch(batch_size=32, drop_remainder=True))

print("shuffle e map aplicadas ")

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])


#copilando o modelo- printar as metricas
model.compile(loss= keras.losses.BinaryCrossentropy(),
              optimizer=tf.optimizers.SGD(lr=0.001),
              metrics=['accuracy',tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.Recall(),
                       tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives()])
model.summary()

print("compilação realizada")

model.fit(train_ds,validation_data=(test_ds),epochs=50)

print('Treinamento feito com sucesso')


results = model.evaluate(test_ds)
print("test loss, test acc:", results)

# Save the entire model as a SavedModel.
model.save('/home/raquel/my_model1')

 



