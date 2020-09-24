
''' This code generates the talk page embeddings for each page.
    
    We use google universal sentence encoder model, where we tokenize the talk page
    into sentences and find the vector representation of each sentence using google
    USE model.

    We then find the average of the sentence representation to get the overall vector
    representation of the talk page and save them in a pickle file for future purposes.
'''

# Tensorflow Version 2.x
import numpy as np
import pandas as pd
import re
import pickle
import text_crawler as tc
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

nltk.download('punkt')

# Load the Google Universal Encoder Model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for generating talk page embeddings')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages_SplToken1.csv',
                                        help='dataset path')
parser.add_argument('--destination', type=str, nargs='?', default='talk_pgs_google_usc_emb.pckl',
                                        help='destination of generated talk page embeddings pckl file')

args = parser.parse_args()


# Read the Dataset
df = pd.read_csv(args.dataset_path)

# Preprocess the Data (Cleaning the Data)
list_sec = []
pages = []
for k in range(len(df)):
    text = df['Talk'][k]
    temp_text = tc.transform(text)
    temp_text = tc.wiki2text(temp_text)
    temp_text = tc.compact(tc.clean(temp_text))
    str1 = ''
    for i in temp_text:
        str1 = str1+i
    list2 = []
    list3 = []
    pat = 'Section::::'
    flag1 = 0
    for i in range(len(temp_text)):
        l = re.findall(pat,temp_text[i])
        if(len(l) > 0):
            flag1 = flag1+1
    flag2= 0
    for i in range(len(temp_text)):
        if(temp_text[i] != ''):
            if(flag2 == flag1):
                a=0
            l = re.findall(pat,temp_text[i])
            if(len(l) > 0):
                flag2 +=1
            if(len(l)== 0):
                list3.append(temp_text[i])
            else:
                if(len(list3) > 0):
                    list2.append(list3)
                    list3 = []
    list2.append(list3)
    list4 = []
    for i in range(len(list2)):
        str1 = ''
        for j in range(len(list2[i])):
            str1 += list2[i][j] + ' '
        list4.append(str1)
    list_sec.append(list4)
    str2 = ''
    for i in range(len(list4)):
        str2 += list4[i]
    pages.append(str2)


# tokenize each page into a list of sentences
list_sent= []
for i in range(len(pages)):
    list2 = []
    for j in range(len(list_sec[i])):
        sent = tokenize.sent_tokenize(list_sec[i][j])
        for k in sent:
            list2.append(k)
    list_sent.append(list2)



''' Compute the google USE embeddings for each sentence and get the mean of all
    the sentences of a talk page as the representation of the talk page.
'''
dict1 = {}
for i in range(len(list_sent)):
    sum1 = np.zeros(512, dtype='float32')
    ab = len(list_sent[i])
    if(ab > 0):
        for j in list_sent[i]:
            emb = embed([j])
            a1 = emb[0].numpy()
            sum1 += a1
        sum1 = sum1/len(list_sent[i])
    dict1[df['Name'][i]] = sum1


# Save the Generated talk page embeddings as a pickle file
with open(args.destination, 'wb') as handle:
    pickle.dump(dict1, handle)



