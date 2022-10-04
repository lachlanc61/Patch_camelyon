import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras


def preplot(config, timg, tlabels):
    
        #plot 12 random images as RGB, including label as true/false 
        fig, ax = plt.subplots(2,6, figsize=(12,5))
        fig.tight_layout(pad=0.1)

        for i,ax in enumerate(ax.flat):
            rand = np.random.randint(config['batchlen'])    
            ax.imshow(timg[rand,:,:,:])
            ax.set_title(bool(tlabels.numpy()[rand]))
            ax.set_axis_off()

        plt.show()
        exit()

def layerplot(config, model, timg, tlabels, odir):
        
    if config['LAYEROUTPLOT']:
        """
        https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
        save outputs of first 16 filters 
            for all conv/pool layers
            for one random image
        """

        #extract layer outputs
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])

        #pick a random image
        rand = np.random.randint(config['batchlen'])   
        img=timg[rand,:,:,:]

        # expand dimensions so that it represents a single 'sample'
        eimg = np.expand_dims(img, axis=0)
        print(eimg.shape)

        #extract feature maps from expanded image
        feature_maps = extractor.predict(eimg)

        #initialise the plots (using mosaic)
        fig, ax = plt.subplot_mosaic("AABCD;AAEFG;HIJKL;MNOPQ", figsize=(16,12))
        fig.tight_layout(pad=1)

        #plot original as RGB, including label as true/false 
        ax["A"].imshow(img)
        ax["A"].set_title(bool(tlabels.numpy()[rand]))
        ax["A"].set_axis_off()

        #iterate through layers, plotting first 16 filter outputs for each
        #save to new file
        #leave original image in place for all

        j=0 #layer index
        for layer in model.layers:
            # skip if not convolutional/pool layer
            if ('conv' not in layer.name) and ('pool' not in layer.name):
                j+=1    #still increment layer index
                continue

            #iterate through first 16 filters
            #   should probably randomise this in future
            i=0   #filter index

            #iterate through mosaic dict for axes
            for key in ax:
                #if looking at original image axis, print the layer and skip
                if key == "A":
                    print(layer.name)
                    print(j)
                else:
                    #plot the feature map
                    # indexing/slicing is weird here, don't fully understand these objects
                    #   [layer][image=0][x,y,filter]
                    #   not sure why it has this format
                    ax[key].imshow(feature_maps[j][0][:,:,i])
                    ax[key].set_axis_off()
                    i+=1

            #save the figure for each layer
            plt.savefig(os.path.join(odir, f'{layer.name}.png'), dpi=300)
            j+=1