import tensorflow as tf
from matplotlib import cm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency


model_path = 'model_path'
images_dir = 'images_path'

# images must be separated into two folders: glaucoma and normal
model = tf.keras.models.load_model(model_path)
category = os.listdir(images_dir)

for c in category:


    if c == 'normal':
        list_images = os.listdir(os.path.join(images_dir,c))
        for i in list_images:
            image_titles = str(i)
            img_path = os.path.join(os.path.join(images_dir,c),i)
            img3 = load_img(img_path, target_size=(224, 224))
            images = np.asarray(np.array(img3))
            X = preprocess_input(images)
            replace2linear = ReplaceToLinear()
            def model_modifier_function(cloned_model):
                cloned_model.layers[-1].activation = tf.keras.activations.linear

            score = CategoricalScore([1])

            saliency = Gradcam(model,
                        model_modifier=replace2linear,
                        clone=True)

            cam = saliency(score, X)
            ax = plt.plot(nrows=1, ncols=1, figsize=(12, 4))
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            plt.imshow(images)
            plt.imshow(heatmap, cmap='jet', alpha=0.5) 
            plt.axis('off')
            plt.tight_layout()
            name = os.path.join(images_dir, c + '-' + i)
            plt.savefig(name)
    

    if c == 'glaucoma':
        list_images = os.listdir(os.path.join(images_dir,c))
        for i in list_images:
            image_titles = str(i)
            img_path = os.path.join(os.path.join(images_dir,c),i)
            img3 = load_img(img_path, target_size=(224, 224))
            images = np.asarray(np.array(img3))
            X = preprocess_input(images)
            replace2linear = ReplaceToLinear()
            def model_modifier_function(cloned_model):
                cloned_model.layers[-1].activation = tf.keras.activations.linear

            score = CategoricalScore([1])

            saliency = Gradcam(model,
                        model_modifier=replace2linear,
                        clone=True)

            cam = saliency(score, X)
            ax = plt.plot(nrows=1, ncols=1, figsize=(12, 4))
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            plt.imshow(images)
            plt.imshow(heatmap, cmap='jet', alpha=0.5) 
            plt.axis('off')
            plt.tight_layout()
            name = os.path.join(images_dir + c + '-' + i)
            plt.savefig(name)


