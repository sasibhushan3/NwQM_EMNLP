''' This code implements the M-BILSTM model (Shen et al., 2019b).

    M-BILSTM model has 2 parts: Text page representation and Image (screenshot) representation. 

    We convert each text page into a list of sentences. We get the representation of a sentence
    by convert each sentence into words and we use 50 dim glove embeddings to get the word vectors.
    This word vectors are passed through an average pooling layer to get the sentence representation.

    These sentence vectors are passed through the  BILSTM to get the page representation.

    We generate the image (screenshot) of a wikipedia page which is then passed through a finetuned
    inceptionV3 model to get the image representation.

    These 2 representations are combined together to get an overall representation of a page and then
    we run a classification layer to classify into 6 Wikipedia classes.
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
import argparse
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import text_crawler as tc
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, AveragePooling1D
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')


# MAX_SENT_LENGTH is max number of words in a sentence
MAX_NB_WORDS = 20000
MAX_SENT_LENGTH = 50
MAX_SENTS = 256
EMBEDDING_DIM = 50


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for M-BILSTM model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                    help='dataset path')
parser.add_argument('--image_embed_path', type=str, nargs='?', default='finetuned_inceptionv3_embeddings.pckl',
                    help='path of generated image embeddings pckl file')
parser.add_argument('--glove_embed', type=str, nargs='?', default='glove.6B.50d.txt',
                    help='path of Glove Embeddings')
parser.add_argument('--num_epoch', type=int, nargs='?', default=50,
                    help='Number of epochs for M-BILSTM model')
parser.add_argument('--batch_size', type=int, nargs='?', default=16,
                    help='Training batch size for M-BILSTM model')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                    help='Learning rate for M-BILSTM model')
args = parser.parse_args()


# Read the Dataset
df = pd.read_csv(args.dataset_path)


# Preprocess the data
# Convert each page into list of Sentences 
pages = []
texts = []
sent_list = []
for i in range(len(df)):
    page_sentences = []
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
            for sent in sentences:
                page_sentences.append(sent)
    sent_list.append(page_sentences)


''' This data array contains the word index of every word present in the page.
    and each page is represented as a list of sentences
'''
data = np.zeros((len(df),MAX_SENTS, MAX_SENT_LENGTH), dtype='float32')


# Fit all the scanned data into the tokenizer
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS,oov_token = True)
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index


# Generate the word index of each word using the tokenizer
for i, sentences in enumerate(sent_list):
    for m, sent in enumerate(sentences):
        if m < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for i_ , word in enumerate(wordTokens):
                if(k < MAX_SENT_LENGTH):
                    if word in tokenizer.word_index:
                        if(tokenizer.word_index[word] < MAX_NB_WORDS):
                            data[i, m, k] = tokenizer.word_index[word]
                            k = k + 1


# Load all the 100 dim Glove Embeddings to a Dictionary
embeddings_index = {}
f = open(args.glove_embed,'r')
for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()


# Load the image representations generated from Finetuned InceptionV3 model
with open(args.image_embed_path, 'rb') as g:
    h = pk.load(g)


# Pages which have image embeddings
img_pages = list(h.keys())
all_pages = []
for i in range(len(df)):
    all_pages.append(df['Name'][i])

page_index = []
avail_pages = []
c = 0
for i in all_pages:
    if i in img_pages:
        avail_pages.append(i)
        page_index.append(c)
    c+=1


# Load the image embeddings
image_data= np.zeros((len(avail_pages),2048), dtype='float32')
c=0
for i in avail_pages:
    image_data[c] = h[i]
    c=c+1


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


# Consider only those pages which have image embeddings
temp = []
for i in page_index:
    temp.append(data[i])
data = np.array(temp)

temp = []
for i in page_index:
    temp.append(labels[i])
labels = np.array(temp)


# Divide the data into train, val and test data
x_train = data[:20000]
x_train_img = image_data[:20000]
y_train = labels[:20000]
x_val = data[20000:24000]
x_val_img = image_data[20000:24000]
y_val = labels[20000:24000]
y_test = labels[24000:30000]
x_test = data[24000:30000]
x_test_img = image_data[24000:30000]



# Build M-BILSTM model
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

sentence_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
embedded_sequences = K.reshape(embedded_sequences, (-1, 50, 50))
pool_out = AveragePooling1D(pool_size=(3), strides=None, padding='valid',
                            data_format='channels_last')(embedded_sequences)
pool_out = K.reshape(pool_out, (-1, 256, 16*50))

l_lstm1 = Bidirectional(LSTM(1, batch_input_shape=(None, 256, 16*50),return_sequences=True))(pool_out)
l_lstm = K.reshape(l_lstm1, (-1, 512))

img_inp = Input(shape=(2048,))
out1 = concatenate([l_lstm,img_inp],axis=-1)
out2 = Dropout(rate = 0.5, noise_shape=None, seed=None)(out1)

preds = Dense(6, activation='softmax')(out2)
optim = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = Model([sentence_input,img_inp], preds)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optim,metrics=['accuracy'])

print("model fitting - M-BILSTM Model")
model.fit([x_train,x_train_img], y_train, nb_epoch=args.num_epoch, batch_size=args.batch_size,
          validation_data=([x_val,x_val_img], y_val))



# Predict and Evaluate the model
evaluate = model.evaluate([x_test,x_test_img],y_test)
y_pred = model.predict([x_test,x_test_img])
y_classes = y_pred.argmax(axis=-1)

print("Accuracy obtained using M-BILSTM model is : ",round(evaluate[1]*100,2))
print("Confusion Matrix of the results obtained using M-BILSTM model is :")
print(confusion_matrix(y_test,y_classes))



