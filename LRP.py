import os
from random import random
from typing import Tuple
import numpy as np
from sklearn import utils
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import math
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Dense
from PIL import Image
from keras.losses import binary_crossentropy, categorical_crossentropy
import time
from multiprocessing import Process
import innvestigate
import innvestigate.utils
import imp


######### GPU CONFIG #########
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


model_path = 'model_path'
images_dir = 'images_path'
destiny_path = 'destiny_path'

loaded_model = keras.models.load_model(model_path)




base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils",os.path.join(base_dir,"utils.py"))
model = innvestigate.utils.model_wo_softmax(loaded_model)

for layer in model.layers:
    layer.name = layer.name + str("_tt")
analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat",model,**{"epsilon": 1})
#'lrp.alpha_beta', 'lrp.alpha_2_beta_1'
category =  os.listdir(images_dir )





for c in category:
    
    pasta_origem = os.path.join(images_dir ,c)
    listimages = os.listdir(os.path.join(images_dir ,c))
    
    for i in listimages:
        
        img_path = os.path.join(os.path.join(images_dir ,c),i)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        prediction = model.predict(img_preprocessed, batch_size=1)
        indice = np.argmax(prediction[0])
        
        if indice == 0:
            predicao = 'glaucoma'
        else:
            predicao = 'normal'
        
        if predicao == c:
            a = 'correto'
        else:
            a = 'erro'
            
        if c == 'glaucoma':
            if predicao == 'glaucoma':
                tc = 'TP'
            else:
                tc = 'FN'
        else:
            if predicao == 'normal':
                    tc = 'TN'
            else:
                tc = 'FP'


        y = utils.load_image(os.path.join(os.path.join(images_dir ,c),i),224)
        plt.imshow(y/255)
        plt.axis("off")

        x = preprocess_input(y[None])

        # Apply analyzer w.r.t. maximum activated output-neuron
        a = analyzer.analyze(x)

        # Aggregate along color channels and normalize to [-1, 1]
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))
        
        # Plot
        plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
        plt.axis("off")
        name = os.path.join(destiny_path + tc + i)
        plt.savefig(name)

