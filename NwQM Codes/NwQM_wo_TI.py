''' This code implements the NwQM-w/oTI model. (NwQM without talk pages and images)
 
    We generate the finetuned bert embeddings for each section of a page, which is
    then passed through a HAN to get the Text page representation.

    These representation is then passed through a classification layer to classify
    into 6 Wikipedia classes.
                            (FA, GA, B, C, Start, Stub)
'''
# Tensorflow Version 1.x
import numpy as np
import pandas as pd
import pickle as pk
import re
import sys
import os
import argparse
from sklearn.metrics import confusion_matrix
from tensorflow import norm
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import  GRU, Bidirectional, TimeDistributed, concatenate
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SECS = 16
EMBED_DIM = 768

# Attention Layer for Hierarchical Attention Network for text page representation
class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1].value, self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_whts = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for NwQM-w/oTI model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--text_embed_path', type=str, nargs='?', default='finetuned_bert_embeddings.pckl',
                                        help='path of generated text embeddings pckl file')
parser.add_argument('--num_epoch', type=int, nargs='?', default=30,
                                        help='Number of epochs for NwQM-w/oTI model')
parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                                        help='Training batch size for NwQM-w/oTI model')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                                        help='Learning rate for NwQM-w/oTI model')
args = parser.parse_args()


# Read the Dataset
df = pd.read_csv(args.dataset_path)


# Load the generated finetuned bert embeddings of each each page
data = np.zeros((len(df),MAX_SECS,EMBED_DIM), dtype='float32')
with open(args.text_embed_path, 'rb') as g:
    h = pk.load(g)

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


# Build NwQM-w/oTI model
page_input = Input(shape=(MAX_SECS, EMBED_DIM), dtype='float32')
l_lstm_sec = Bidirectional(GRU(100, return_sequences=True))(page_input)
l_att_sec = AttLayer(100)(l_lstm_sec)

preds = Dense(6, activation='softmax')(l_att_sec)

model = Model(page_input, preds)
optim = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optim, metrics=['acc'])

print("model fitting - NwQM-w/oTI Model")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=args.num_epoch, batch_size=args.batch_size)

# Predict and Evaluate the model
evaluate = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
y_classes = y_pred.argmax(axis=-1)

print("Accuracy obtained using NwQM-w/oTI model is : ",round(evaluate[1]*100,2))
print("Confusion Matrix of the results obtained using NwQM-w/oTI model is :")
print(confusion_matrix(y_test,y_classes))

