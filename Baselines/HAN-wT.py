
''' This code implements the Hierarchical Attention Network (HAN) model with talk page representations
    for Wikipedia Pages.

    Wikipedia pages have a 4 level Hierarchy : words --> sentence (words combine to from a sentence),
    sentences --> paragraph, paragraphs --> section, sections --> page.

    We use glove embeddings for word representations. We then use this HAN to get sentence
    representation, followed by paragraph representation, section representation and finally get 
    the page representation. 

    Every Wikipedia page has a talk page, So we use Google Universal Sentence Encoder (Google USE)
    model to get the talk page representation.

    We then concatenate the page representation from HAN and talk page represntation from Google
    USC and run a classification layer to classify into 6 Wikipedia classes.
                                (FA, GA, B, C, Start, Stub)
'''
# Tensorflow Version 1.x
import numpy as np
import pandas as pd
import pickle
import re
import sys
import os
import fnmatch
import nltk
import argparse
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import text_crawler as tc
from sklearn.metrics import confusion_matrix
from tensorflow import norm
from tensorflow.keras.layers import multiply
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Embedding, Dropout, GRU, Bidirectional, TimeDistributed, concatenate

nltk.download('punkt')

# MAX_SENT_LENGTH is max number of words in a sentence
MAX_SENT_LENGTH = 50
MAX_SENTS = 7
MAX_SECS = 16
MAX_PARAS = 5
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

# Attention Layer for Hierarchical Attention Network
class AttLayer(Layer):
        def __init__(self, attention_dim):
                self.init = initializers.get('normal')
                self.supports_masking = True
                self.attention_dim = attention_dim
                super(AttLayer, self).__init__()

        def build(self, input_shape):
                assert len(input_shape) == 3
                self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
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
                        # Cast the mask to floatX to avoid float64 upcasting in theano
                        ait *= K.cast(mask, K.floatx())
                ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
                ait = K.expand_dims(ait)
                weighted_input = x * ait
                output = K.sum(weighted_input, axis=1)

                return output

        def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1])




# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for Hierarchical Attention Network with talk model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--glove_embed', type=str, nargs='?', default='glove.6B.100d.txt',
                                        help='path of Glove Embeddings')
parser.add_argument('--talk_embed_path', type=str, nargs='?', default='talkpage_embeddings.pckl',
                                        help='path of generated talk page embeddings pckl file')
parser.add_argument('--num_epoch', type=int, nargs='?', default=10,
                                        help='Number of epochs for HAN with talk model')
parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                                        help='Training batch size for HAN with talk model')
args = parser.parse_args()


# Read the Dataset
df = pd.read_csv(args.dataset_path)
df = df[0:30000]


# Preprocess the data
''' Convert each page into list of sections, each section into 
    list of  paragraphs and each paragraph into list of sentences.
'''
pages = []
texts = []
list_pages = []
for i in range(len(df)):
    text = df['Text'][i]
    temp_text = tc.transform(text)
    temp_text = tc.wiki2text(temp_text)
    temp_text = tc.compact(tc.clean(temp_text))
    str1 = ''
    for i in temp_text:
        str1 = str1+i
    texts.append(str1)
    list2 = []
    list3 = []
    pat = 'Section::::'
    flag1 = 0
    for i in range(len(temp_text)):
        l = re.findall(pat,temp_text[i])
        if(len(l) > 0):
            flag1 = flag1+1

    flag2= 0
    for j in range(len(temp_text)):
        if(temp_text[j] != ''):
            if(flag2 == flag1):
                a=0
            l = re.findall(pat,temp_text[j])
            if(len(l) > 0):
                flag2 +=1
            if(len(l)== 0):
                list3.append(temp_text[j])
            else:
                if(len(list3) > 0):
                    list2.append(list3)
                    list3 = []
    list2.append(list3)
    list_sec = []
    for k in range(len(list2)):
        list_parag = []
        for m in range(len(list2[k])):
            para = list2[k][m]
            sentences = tokenize.sent_tokenize(para)
            list_parag.append(sentences)
        list_sec.append(list_parag)
    list_pages.append(list_sec)


# Fit all the scanned data into the tokenizer
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS,oov_token = True)
tokenizer.fit_on_texts(texts)


# This data array contains the word index of every word present in the page in the 4 level Hierarchical format.
data = np.zeros((len(df), MAX_SECS, MAX_PARAS, MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

# Generate the word index of each word using the tokenizer
for i, sections in enumerate(list_pages):
        for j, paras in enumerate(sections):
            for l, sentences in enumerate(paras):
                for m, sent in enumerate(sentences):
                    if j < MAX_SECS:
                        if l < MAX_PARAS:
                            if m < MAX_SENTS:
                                wordTokens = text_to_word_sequence(sent)
                                k = 0
                                for _, word in enumerate(wordTokens):
                                    if(k < MAX_SENT_LENGTH):
                                        if word in tokenizer.word_index:
                                        # print(i,j,l,m,k)
                                            if(tokenizer.word_index[word] < MAX_NB_WORDS):
                                                data[i, j,l,m, k] = tokenizer.word_index[word]
                                                k = k + 1

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


# Load the talk page representations generated from Google USE model
talk_data = np.zeros((len(df),512),dtype = 'float32')
with open(args.talk_embed_path, 'rb') as g:
    h = pk.load(g)
for i in range(len(df)):
    name = df['Name'][i]
    talk_data[i] = h[name]




x_train = data[:20000]
x_talk_train = talk_data[:20000]
y_train = labels[:20000]
x_val = data[20000:24000]
x_talk_val = talk_data[20000:24000]
y_val = labels[20000:24000]
x_test = data[24000:30000]
x_talk_test = talk_data[24000:30000]
y_test = labels[24000:30000]

# Load all the 100 dim Glove Embeddings to a Dictionary
embeddings_index = {}
f = open(args.glove_embed,'r')
for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()

word_index = tokenizer.word_index



# Building Hierachical Attention Network Model
# Create Embedding Layer for generating word Embeddings
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                            weights=[embedding_matrix], input_length=MAX_SENT_LENGTH,
                            trainable=True, mask_zero=True)

# Generate Sentence Representation from words
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)

# Generate Paragraph Representation from sentences
sentEncoder = Model(sentence_input, l_att)
para_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
para_encoder = TimeDistributed(sentEncoder)(para_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(para_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)

# Generate Section Representation from paragraphs
paraEncoder = Model(para_input, l_att_sent)
sec_input = Input(shape=(MAX_PARAS,MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
sec_encoder = TimeDistributed(paraEncoder)(sec_input)
l_lstm_para = Bidirectional(GRU(100, return_sequences=True))(sec_encoder)
l_att_para = AttLayer(100)(l_lstm_para)

# Generate Page Representation from sections
secEncoder = Model(sec_input, l_att_para)
page_input = Input(shape=(MAX_SECS,MAX_PARAS,MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
page_encoder = TimeDistributed(secEncoder)(page_input)
l_lstm_sec = Bidirectional(GRU(100, return_sequences=True))(page_encoder)
l_att_sec = AttLayer(100)(l_lstm_sec)

# Add the Talk page represntation obtained from Google USE model
talk_inp = Input(shape=(512,))
talk_rep = Dense(200, activation='relu')(talk_inp)

# Combine both representations and classify into 6 class model
norm1 = norm(l_att_sec-talk_rep,ord =2,keepdims = True,axis = -1)
final_rep = concatenate([l_att_sec,talk_rep,norm1],axis=-1)
preds = Dense(6, activation='softmax')(final_rep)
model = Model([page_input,talk_inp], preds)

model.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam', metrics=['acc'])

print("model fitting - Hierachical attention network with talk page representations")
model.fit([x_train,x_talk_train], y_train, validation_data=([x_val,x_talk_val], y_val),
          nb_epoch=args.num_epoch, batch_size=args.batch_size)



# Predict and Evaluate the model
evaluate = model.evaluate([x_test,x_talk_test],y_test)
y_pred = model.predict([x_test,x_talk_test])
y_classes = y_pred.argmax(axis=-1)

print("Accuracy obtained using Hierachical Attention Network model is : ",round(evaluate[1]*100,2))
print("Confusion Matrix of the results obtained using Hierachical Attention Network model is :")
print(confusion_matrix(y_test,y_classes))
