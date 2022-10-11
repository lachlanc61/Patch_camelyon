
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import Sequential 

# for tensorflow keras
from classification_models.tfkeras import Classifiers

import src.utils as utils


def build(config):

    
    #https://www.kaggle.com/competitions/histopathologic-cancer-detection/discussion/83760


    #https://stackoverflow.com/questions/66679241/import-resnext-into-keras
    #resnext50 in keras
    #ResNeXt50, preprocess_input = Classifiers.get('resnext50')
    #base_model = ResNeXt50(include_top = False, input_shape=(96, 96, 3), weights='imagenet')

    """
    base_model = keras.applications.resnet50.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(96, 96, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.
    """

    #https://keras.io/api/applications/efficientnet/
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3),
    )

    base_model.trainable = False


    inputs = keras.Input(shape=(96, 96, 3))
    
    x = base_model(inputs, training=False)

    #https://keras.io/guides/transfer_learning/
    x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu', name="dense1")(x)
    x = keras.layers.Dropout(0.3)(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1, activation = 'sigmoid', name="output")(x)

    """
    outputs = Sequential(
        [
            keras.layers.AveragePooling2D(pool_size = 3),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', name="dense1"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation = 'sigmoid', name="output"),
        ], name="tail1"
    )
    """
    model = keras.Model(inputs, outputs)



    """
    model = Sequential(
        [
            keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (96, 96, 3), name="b1_conv1"),
            keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', name="b1_conv2"),
            keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', name="b1_conv3"),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size = 3),

            keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', name="b2_conv1"),
            keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', name="b2_conv2"),
            keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', name="b2_conv3"),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size = 3),

            keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', name="b3_conv1"),
            keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', name="b3_conv2"),
            keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', name="b3_conv3"),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size = 3),

            keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu', name="b4_conv1"),
            keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu', name="b4_conv2"),
            keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu', name="b4_conv3"),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', name="dense1"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation = 'sigmoid', name="output"),
        ], name = "model3" 
    )  
    """

    #view model summary
    model.summary()

    #compile the model
    #   acc = track accuracy vs train/val sets
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['acc'],
    )

    return model

def train(model, dtrain, dval, config, checkpoint_path):

    #Callbacks
    #   (special utilities executed during training)
    # https://blog.paperspace.com/tensorflow-callbacks/

    #stop refinement early if val stops improving
    #https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
    stopcond=keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['stoplim'])


    #save model during training every 5 batches
    # save best model only based on val_loss

    #ok i am stumped on saving best model
    #   get warning "Can save best model only with val_acc available, skipping"
    #   looks like val_acc not making it into model.metrics somehow

    #BUT val_loss works for early stoping callback. both are available in history
    #I guess just save every 5 for now.... creates a pile of checkpoints, 
    #       will miss very best one unless save every epoch

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                #  save_best_only=True,
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    save_freq=1*config['savesteps'])

    #TRAIN/LOAD

    #if training, do it
    #else load model from checkpoint variable (manually set)

    if config['DOTRAIN']:
        #run the fit, saving params to fitlog 
        #   epochs = N. cycles
        #   validation data is tested each epoch
        fitlog = model.fit(
            dtrain,
            epochs=config['nepochs'],
            validation_data = dval,
            validation_freq = 1,
            callbacks=[stopcond,cp_callback],
            verbose=1
        )

        print(model.metrics_names)

        #demonstrating val_acc is present
        for key in fitlog.history:
            print(key)
    
        #extract metrics for plotting
        tacc = fitlog.history['acc']
        tloss = fitlog.history['loss']
        vacc = fitlog.history['val_acc']
        vloss = fitlog.history['val_loss']


        #plot fit progress against train and validation data
        if config['POSTPLOT']:
            fig, ax = plt.subplots(1,2, figsize=(12,6))
            fig.tight_layout(pad=2)

            epochs = range(1, len(tacc) + 1)
            ax[0].plot(epochs, tacc, 'r', label='Training accuracy')
            ax[0].plot(epochs, vacc, 'b', label='Validation accuracy')
            ax[0].set_title('Accuracy')
            ax[0].legend()

            ax[1].plot(epochs, tloss, 'r', label='Training loss')
            ax[1].plot(epochs, vloss, 'b', label='Validation loss')
            ax[1].set_title('Loss')
            ax[1].legend()

            plt.show()
        else: #if not training
            #load from checkpoint variable
            model.load_weights(config['cptoload'])
            tloss, tacc = model.evaluate(dtrain, verbose=2)
            vloss, vacc = model.evaluate(dval, verbose=2)

            print("MODEL LOAD SUCCESSFUL\n"
            f'checkpoint: {config["cptoload"]}\n'
            f'------------------------\n'
            f'train loss: {tloss:>9.3f}\n'
            f'train acc : {tacc:>9.3f}\n'
            f'------------------------\n'
            f'val loss  : {vloss:>9.3f}\n'
            f'val acc   : {vacc:>9.3f}\n')
            f'------------------------\n'

        #  calc and print test result
        #   prefer not to see this till later
        if False:
            eval = model.evaluate(testimg, testlabels)
            print("EVAL\n"
            f'test loss: {eval[0]}\n'
            f'test acc:  {eval[1]}\n')
    return model, fitlog