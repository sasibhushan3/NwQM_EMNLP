''' This code generates the finetuned InceptionV3 embeddings for each page from
    the InceptionV3 model which we finetuned.
'''
import numpy as np
import pandas as pd
import os
import pickle
import itertools
import time
import random
import argparse
import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.preprocessing import LabelEncoder


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for Finetuned InceptionV3 model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--images_path', type=str, nargs='?', default='wiki_images/',
                                        help='path of the folder containing images of the pages')
parser.add_argument('--model_path', type=str, nargs='?', default='finetuned_inceptionv3_model.h5',
                                        help='Path of the Finetuned InceptionV3 Model')
parser.add_argument('--destination', type=str, nargs='?', default='finetuned_inceptionv3_embeddings.pckl',
                                        help='Destination for saving Finetuned InceptionV3 Embeddings')

args = parser.parse_args()


# Read the Dataset
df = pd.read_csv(args.dataset_path)


# List of all pages (image names) present in the dataset
page_names = []
for i in range(len(df)):
    page_names.append(df['Name'][i])


# names of all image files present the folder
image_names = os.listdir(args.images_path)


# Keep only those images which are present in both the folder and the dataset
c=0
dataset_indices = []
image_labels = []
avail_images = []
for i in page_names:
    j = i+'.jpg'
    if j in image_names:
        dataset_indices.append(c)
        image_labels.append(df['Label'][c])
        avail_images.append(df['Name'][c])
    c=c+1


# Convert the labels of each page in Numerical Format
labels = []
for i in range(len(dataset_indices)):
    a = image_labels[i]
    if(a == 'FA'):
        labels.append(0)
    if(a == 'GA'):
        labels.append(1)
    if(a == 'B'):
        labels.append(2)
    if(a == 'C'):
        labels.append(3)
    if(a == 'Start'):
        labels.append(4)
    if(a == 'Stub'):
        labels.append(5)


# generate the image path and its label as a tuple for processing
proc = []
for i in range(len(avail_images)):
    j = args.images_path + avail_images[i] + '.jpg'
    k = labels[i]
    proc.append((j,k))


# Load the Finetuned InceptionV3 Model
inception = load_model(args.model_path)


# The layer containing the InceptionV3 Model Output
inception_layer = K.function([inception.layers[0].input],
                             [inception.layers[-2].output])


# Generate the Finetuned InceptionV3 Embeddings for each page
data = {}
for i in len(proc):
    img_path = proc[i][0]
    try:
        img = image.load_img(img_path,target_size =(500,500))
    except:
        print('loading error')
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)

    layer_output = inception_layer([x])[0][0]
    name = avail_images[i]
    data[name] = layer_output


# Save the Embeddings
with open(args.destination,'wb') as h:
    pickle.dump(data,h)
