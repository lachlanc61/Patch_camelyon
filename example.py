"""
Example code
https://medium.com/analytics-vidhya/deep-learning-tutorial-patch-camelyon-data-set-d0da9034550e
  6 conv2D layers + 6 maxpool layers
  3 dense layers
  gets ~0.75, 150ms/step on GPU
"""
#Importing the necessary libraries

from tensorflow import keras
import tensorflow as tf
import os,datetime
import tensorflow_datasets as tfds
import sys

from keras import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras import layers
from keras import optimizers
from keras import regularizers

import matplotlib.pyplot as plt

print(sys.version)
gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)
print(gpu_available)

#Loading the data from tensorflow_datasets
df, info = tfds.load('patch_camelyon', with_info = True, as_supervised = True)

#Getting the train, validation and test data
train_data = df['train']
valid_data = df['validation']
test_data = df['test']

#A function to help scale the images
def preprocess(image, labels):
  image = tf.cast(image, tf.float32)
  image /= 255.
  return image, labels

#Applying the preprocess function we the use of map() method
train_data = train_data.map(preprocess)
valid_data = valid_data.map(preprocess)
test_data = test_data.map(preprocess)

#Shuffling the train_data
buffer_size = 3000
train_data = train_data.shuffle(buffer_size)

#Batching and prefetching
batch_size = 128
train_data = train_data.batch(batch_size).prefetch(1)
valid_data = valid_data.batch(batch_size).prefetch(1)
test_data = test_data.batch(batch_size).prefetch(1)

#Seperating image and label into different variables
train_images, train_labels = next(iter(train_data))
valid_images, valid_labels = next(iter(valid_data))
test_images, test_labels  = next(iter(test_data))

#Checking the label shape
print(valid_labels.shape)

#Checking the image shape
print(train_images.shape)


"""
MODEL HERE
"""

model = Sequential([
                    Conv2D(256, 3,padding='same', kernel_initializer='he_uniform', activation='relu', input_shape = [96, 96, 3]),
                    MaxPooling2D(2),
                    Conv2D(256, 3,padding='same', kernel_initializer='he_uniform',activation='relu',),
                    MaxPooling2D(2),
                    Conv2D(512, 3,padding='same',kernel_initializer='he_uniform',activation='relu',),
                    MaxPooling2D(2),
                    Conv2D(512, 3,padding='same',kernel_initializer='he_uniform',activation='relu',),
                    MaxPooling2D(2),
                    Conv2D(1024, 3,padding='same', kernel_initializer='he_uniform',activation='relu',),
                    MaxPooling2D(2),
                    Conv2D(1024, 3,padding='same', kernel_initializer='he_uniform',activation='relu',),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(1028,kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(512,kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
                    Dense(1, activation = 'sigmoid'),
                    ])
 
model.summary()


#Compiling our model
model.compile(optimizer= optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

#Callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#Fitting our model
history = model.fit( train_images, train_labels, epochs = 100, callbacks=[early_stopping_cb], validation_data = (valid_images, valid_labels), verbose=2)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.evaluate(test_images, test_labels)

print("CLEAN EXIT")