
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import Sequential 

import src.utils as utils


def build(config):
    #we are overfitting pretty heavily, try regularisation
    l2reg=tf.keras.regularizers.L2(config['lamda'])
    #kernel_regularizer=l2reg
    #not great, slow and not a big improvement, leave it out for now


    """
    MODEL HERE

    convolution layer (Nfilters,Npx,...)
        basically a set of N filters acting on a window of MxM px slid across whole image
        output shape x,y,Nfilters
    
    max pooling layer (Npx)
        sliding Npx x Npx window
        outputs max value within window
        effectively downsamples image retaining max
        https://www.youtube.com/watch?v=ZjM_XQa5s6s

    use relu function instead of signmoid for all but output layer
    basically max(0,val)
    = passthrough if above 0
        more responsive across entire range compared to sigmoid

    """

    #Initialise basic TF model
    """
    Simpler, faster model for testing, still gets 0.7-0.75 most of the time

    model = Sequential(
        [
            keras.layers.Conv2D(64,3, padding='same', activation='relu', input_shape=[96, 96, 3], name="block1_conv1"),
            keras.layers.MaxPooling2D(2, name="block1_pool"),
            keras.layers.Conv2D(32,3, padding='same', activation='relu', name="block2_conv1"),
            keras.layers.MaxPooling2D(2, name="block2_pool"),
            keras.layers.Conv2D(16,3, padding='same', activation='relu', name="block3_conv1"),
            keras.layers.MaxPooling2D(2, name="block3_pool"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu', name="dense1"),
            keras.layers.Dense(128, activation='relu', name="dense2"),
            keras.layers.Dense(56, activation='relu', name="dense3"),
            keras.layers.Dense(12, activation='relu', name="dense4"), 
            keras.layers.Dense(1, activation='sigmoid', name="predictions") 
        ], name = "my_model" 
    )   
    

    
    #gets ~0.75 fairly consistently - slow, large
    model = Sequential(
        [
            keras.layers.Conv2D(256,3, padding='same', activation='relu', input_shape=[96, 96, 3]),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(1024,3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(256,3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(56, activation='relu'),
            keras.layers.Dense(12, activation='relu'), 
            keras.layers.Dense(1, activation='sigmoid') 
        ], name = "my_model" 
    )   
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
    

    #view model summary
    model.summary()

    #compile the model
    #   acc = track accuracy vs train/val sets
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.0005),
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