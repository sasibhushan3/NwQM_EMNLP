''' This code implements the Talk model.

    We use Google Universal Sentence Encoder (Google USE) model to get the Talk page Representation
    of a page.

    This is passed through a classification layer to classify into 6 Wikipedia classes.
                            (FA, GA, B, C, Start, Stub)
'''
# Tensorflow Version 1.x
import numpy as np
import pandas as pd
import pickle
import argparse
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for Talk model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--talk_embed_path', type=str, nargs='?', default='talkpage_embeddings.pckl',
                                        help='path of generated talk page embeddings pckl file')
parser.add_argument('--num_epoch', type=int, nargs='?', default=30,
                                        help='Number of epochs for Talk model')
parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                                        help='Training batch size for Talk model')
args = parser.parse_args()

# Read the Dataset
df = pd.read_csv(args.dataset_path)


# Load the talk page representations generated from Google USE model
with open(args.talk_embed_path, 'rb') as g:
    h = pickle.load(g)


data = np.zeros((len(df),512), dtype='float32')
for i in range(len(df)):
    name = df['Name'][i]
    data[i] = h[name]


# Convert the labels of each page in Numerical Format
labels = []
for i in range(len(df)):
    a = df['Label'][i]
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
labels = np.asarray(labels)


# Divide the data into train, val and test data
x_train = data[:20000]
x_val = data[20000:24000]
x_test = data[24000:30000]
y_train = labels[:20000]
y_val = labels[20000:24000]
y_test = labels[24000:30000]


# Build Talk Model
talk_inp = Input(shape=(512,))
x2 = Dense(6, activation='softmax')(talk_inp)
model = Model([talk_inp], x2)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.fit([x_train], y_train, validation_data=([x_val], y_val),
          nb_epoch=args.num_epoch, batch_size=args.batch_size)



# Evaluate the model
evaluate = model.evaluate([x_test],y_test)
print("Accuracy obtained using Talk model is : ",round(evaluate[1]*100,2))

