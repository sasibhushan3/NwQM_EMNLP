''' This code Finetunes the InceptionV3 model on the Wikipedia data for getting better
    representation of the page.
 
    We generate the images (screenshot) of each page and the image is scaled.

    We then generate a generator object with a predefined variable batch_size
    which contains the image and the label of each page as batches of size batch_size.

    It is then passed to the inceptionV3 model which is finetuned for 6 class 
    classification to classify into 6 wikipedia classes and save the Finetuned model.
                            (FA, GA, B, C, Start, Stub)
'''
import numpy as np
import pandas as pd
import os
import pickle
import itertools
import time
import random
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.preprocessing import LabelEncoder


''' Given a document return X, y

    X is scaled to [0, 1] and consists of all images contained in document.
    y is given an integer encoding.

    Returns as a Generator Object of documents(pages) with each element of
    size batch_size.
'''
def get_features_label(documents, batch_size, return_labels=True):
    start = 0
    labels = []
    images = []
    while True:
        del images, labels
        
        if(start + batch_size > len(documents)):
            random.shuffle(documents)
            start = 0
     
        images = []
        labels = []

        for document in documents[start: start + batch_size]:
            label = document[1]
            img_path = document[0]
            try:
                img = image.load_img(img_path,target_size =(500,500))
            except:
                print('loading error')

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            x = x[0]
            images.append(x)
            labels.append(label)

        start += batch_size
        yield np.array(images), np.array(labels)


''' Creates the Finetuned Inception Model with keeping all layers in 
    the inception model as trainable and adds a GlobalAveragePooling2D 
    layer above the inception model and finally a 6 class classificafication layer.
'''
def create_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             input_shape = (500,500,3))    
    # add a global spatial average pooling layer
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(6, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)    
    return model


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for Finetuned InceptionV3 model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages_SplToken1.csv',
                                        help='dataset path')
parser.add_argument('--images_path', type=str, nargs='?', default='wiki_images/',
                                        help='path of the folder containing images of the pages')
parser.add_argument('--destination', type=str, nargs='?', default='finetuned_inceptionv3_model.h5',
                                        help='Destination of saving Finetuned InceptionV3 Model')
parser.add_argument('--num_epoch', type=int, nargs='?', default=20,
                                        help='Number of epochs for Finetuned InceptionV3 model')
parser.add_argument('--batch_size', type=int, nargs='?', default=16,
                                        help='Batch size of the Generator Object')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                                        help='Learning rate for Finetuned InceptionV3 model')
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


# Divide into train, val data
train = proc[:20000]
val = proc[20000:24000]


# Process the Data and generate the generator objects for train and val data
train_generator = get_features_label(train, batch_size=args.batch_size)
val_generator = get_features_label(val, batch_size=args.batch_size)


# Create the model
inception = create_model()
callback = keras.callbacks.TensorBoard(
    log_dir='./logs/inception/2/{}'.format(time.time()))


# Make all layers in the InceptionV3 model as trainable
for layer in inception.layers:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

inception.compile(optimizer=Adam(lr=args.learning_rate),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])


# Stpes per epoch (since each element of the generator object contains batch_size number of pages)
train_steps = int(len(train)/args.batch_size)
val_steps = int(len(val)/args.batch_size)


# Train the model
inception.fit_generator(
        generator=train_generator,
        epochs=args.num_epoch,
        steps_per_epoch=train_steps,
        callbacks=[es],
        validation_data=val_generator,
        validation_steps=val_steps
    )


# Save the model
inception.save(args.destination)