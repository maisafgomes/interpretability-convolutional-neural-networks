from datetime import datetime
import os
import shutil
import numpy as np
import tensorflow  as tf
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as img
from keras.applications.resnet50 import preprocess_input
import math
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Dense
from PIL import Image
from keras.losses import binary_crossentropy, categorical_crossentropy
import time
import random

from keras.preprocessing.image import ImageDataGenerator 

def divide(path,database,name,destiny):
    dir = os.path.join(destiny,name)
    os.makedirs(dir)
    classes = os.listdir(os.path.join(path,database))
    print(classes[0])
    for classe in classes:
        if classe == 'glaucoma':
            list_images_glaucoma = os.listdir(os.path.join(path,database,classes[0]))
        else:
            list_images_normal = os.listdir(os.path.join(path,database,classes[1]))
    
    size = min(len(list_images_glaucoma),len(list_images_normal)) 
    glaucoma_select = random.sample(range(0, len(list_images_glaucoma)), size)
    normal_select = random.sample(range(0, len(list_images_normal)), size)

    for p in classes:
        des = os.path.join(destiny,name,p)
        os.makedirs(des)
        if p == 'glaucoma':
            for num in range(len(glaucoma_select)):
                origin = os.path.join(path,database,p,list_images_glaucoma[glaucoma_select[0]])
                shutil.copyfile(origin, os.path.join(des,list_images_glaucoma[glaucoma_select[0]]))
                img = Image.open(os.path.join(des,list_images_glaucoma[glaucoma_select[0]]))
                img = img.resize((224,224))
                img.save(os.path.join(des,list_images_glaucoma[glaucoma_select[0]]))
                glaucoma_select.pop(0)
        if p == 'normal':
            for num in range(len(normal_select)):
                origin = os.path.join(path,database,p,list_images_normal[normal_select[0]])
                shutil.copyfile(origin, os.path.join(des,list_images_normal[normal_select[0]]))
                img = Image.open(os.path.join(des,list_images_normal[normal_select[0]]))
                img = img.resize((224,224))
                img.save(os.path.join(des,list_images_normal[normal_select[0]]))
                normal_select.pop(0)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""
###  Variaveis ###
batch = 8
epoch = 200
inputSize = (224,224,3)


def generate_from_paths_and_labels(input_paths, labels, batch_size, input_size= inputSize):
    num_samples = len(input_paths)
    while 1:
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
              lambda x: img.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: img.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs, labels[i:i+batch_size])

def train_dataset(x,y): 
    return generate_from_paths_and_labels(
        input_paths=x,
        labels=y,
        batch_size= batch
    )


## Preparação do DataBase ##
path = '/home/maisa/Área de Trabalho/ARTIGO /pred/DATABASE'
destiny = 'experimentos_2'
database = 'acrima_deleta_clame'
name = datetime.today().strftime('%Y-%m-%d__%H:%M:%S')
name = name + '__' + database
divide(path,database,name,destiny)
datadir =  os.path.join(destiny,name)
classes_dir = ['glaucoma','normal']
num_classes = len(classes_dir)
input,labels = [],[]

for classes in classes_dir:
    class_root = os.path.join(datadir, classes)
    class_id = classes_dir.index(classes)
    print('class name:', classes)
    print('class id:', class_id)
    for path in os.listdir(class_root):
        path = os.path.join(class_root, path)
        input.append(path)
        labels.append(class_id)

# convert to one-hot-vector format
labels = to_categorical(labels, num_classes=num_classes)

# convert to numpy array
input_paths = np.array(input)

# Dividindo o dataset entre teste e treino
train_input_paths, test_input_paths, train_labels, test_labels = train_test_split(input_paths, labels, test_size=0.30, random_state = 40, shuffle = True)

## Implementação da base Resnet50 ##
model_base = keras.applications.ResNet50(
    include_top=False,
    weights= 'imagenet',
    input_shape= inputSize,
    pooling = None
    )

last_layer = model_base.get_layer('activation_49')
x = GlobalAveragePooling2D()(last_layer.output)
x = Dense(2,activation='softmax')(x)
model_final = Model(model_base.input,x)
for layer in model_final.layers:
    layer.trainable=True

file1 = open("sabado_acrima_rim-one_clame.csv", "a")


model = model_final
opt = adam(lr=0.0001, epsilon=1e-11, decay=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

A = train_dataset(train_input_paths,train_labels)
B = train_dataset(test_input_paths,test_labels)

inicio = time.time()
history = model.fit_generator(A,steps_per_epoch = math.ceil(len(train_input_paths)/batch), epochs = epoch, validation_data=B,validation_steps= math.ceil(len(test_input_paths)/batch))
fim = time.time()
    
n = "acriam_clame"+ str(fold_no)

plt.plot(history.history['acc'],color = 'm')
plt.plot(history.history['val_acc'], color = 'g')
plt.title('Acurácia Modelo A')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.ylim(0,1.01)
plt.savefig(n + 'acc')    
plt.clf()


plt.plot(history.history['loss'],color = 'm')
plt.plot(history.history['val_loss'],color = 'g')
plt.title('Perda Modelo A')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.ylim(0,1)
plt.savefig(n + 'loss')
plt.clf()

#Generate generalization metrics
scores = model.evaluate_generator(B,steps=math.ceil(len(test_input_paths)/batch), verbose=0)
print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
model.save("mol"+str(fold_no) + '.h5')
print(((fim - inicio)/60))
 
    