
''' Code to run History Based LSTM model, which extracts the content and meta features
    from given number of revisions (History) of each page and combines these features
    using BiLSTM model to classify into the 6 Wikipedia Quality Classes.
                    (FA, GA, B, C, Start, Stub)
'''

import numpy as np
import pandas as pd
import re
import fnmatch
import pickle
import os
import time
import random
import datetime as dt
import dateutil.parser
import text_crawler as tc
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical

''' This function Extracts featues from last n revisions of each page 

    num_features = 16, if both content and meta features are extracted
                   11, if only content features are extracted

    content features = Article Length, Number of References, Number of Category Links,
        Number of Other Links, Number of citation templates, Number of non-citation templates,
        Number of images, Has an infobox template (Binary), Number of level 2 section headings,
        Number of level 3+ section headings, Information noise score.

    Meta Features = timestamp, minor, revert, comment length
'''
def extract_features(n, pages_list, page_labels, num_features, files_path):
    stopw = set(stopwords.words('english'))
    clean = re.compile('<.*?>')
    clean2 = re.compile('\[\[.*?\]\]')
    clean3 = re.compile('\{\{.*?\}\}')
    tokeniz = RegexpTokenizer(r'\w+')
    array3 = np.zeros([len(pages_list),n,num_features])
    count1 = 0
    count2 = 0
    key_list = ['timestamp', 'user', 'minor', 'comment', 'text']
    avail_page_labels = []
    avail_pages = []
    for i in pages_list:
        f = open(files_path+i,'rb')
        f1 = pickle.load(f)   
        f2 = f1['revision']
        num_rev = len(f2)
        array2 = np.empty([n,num_features])
        flag1 = 0
        if(len(f2)>=n):
            for j in range(n):
                f3=f2[num_rev-n+j]
                flag2 = 0
                flag3 = 0
                loo = f3.keys()
                for k in key_list:
                    if k in loo:
                        flag3 += 1
                if(flag3 < 5):
                    flag2 = 1
                    break  

                # time stamp              
                timestamp = f3['timestamp']
                dat = dateutil.parser.parse(timestamp)
                timestamp2 = int(time.mktime(dat.timetuple()))
                
                flag1 = 0
                count2 = count2+1
                if 'id' not in f3['user'].keys():
                    flag1 = 1
                    break
                userid = f3['user']['id']
                
                if 'text' not in f3.keys():
                    flag1 = 1
                    break
                    
                size = len(f3['text'])
                
                # minor
                if(f3['minor'] == False):
                    minor2 = 0
                else:
                    minor2 = 1
                
                # comment length    
                comm_len = len(f3['comment'])
                
                str1 = f3['comment']
                str2 = 'revert'
                str3 = 'rv'

                # revert
                if(str1.find(str2) != -1):
                    revert = 1
                elif(str1.find(str3) != -1):
                    revert = 1
                else:
                    revert = 0
                
                # Article Size
                text = f3['text']
                art_length = size
                
                # Number of References
                pat1 = '<ref>'
                l1 = re.findall(pat1,text)
                num_ref = len(l1)
                
                pat2= '\[\[.*?\]\]'
                l2 = re.findall(pat2,text)
                co = 0

                # Number of Category Links
                for im in l2:
                    if(im.find('Category')!=-1):
                        co = co+1
                cate_links = co
                
                # Number of Other Links
                oth_links = len(l2) - cate_links
                
                pat3 = '\{\{.*?\}\}'
                l3 = re.findall(pat3,text)

                # Number of citation templates
                num_cite_temp = 0
                for im in l3:
                    if(im.find('cite')!=-1):
                        num_cite_temp = num_cite_temp +1

                # Number of non citation templates
                num_non_cit_temp = len(l3) - num_cite_temp
                
                # Number of Images
                mystr = text
                jpg=jpeg=png=bmp=gif=svg=0
                jpg=mystr.count('.jpg' or '.JPG')
                jpeg=mystr.count('.jpeg' or  '.JPEG')
                svg=mystr.count('.svg' or '.SVG')
                gif=mystr.count('.gif' or '.GIF')
                png=mystr.count('.png' or '.PNG')
                bmp=mystr.count('.bmp' or '.BMP')
                num_images=jpg+jpeg+svg+gif+png+bmp
                
                # Has an infobox template (Binary)
                inf_tmp = 0
                for im in l3:
                    if(im.find('Infobox') != -1):
                        inf_tmp = 1
                    elif(im.find('infobox')!= -1):
                        inf_tmp = 1
                
                pat4 = '==.*?=='
                l6 = re.findall(pat4,text)
                pat5 = '===.*?==='
                l7 = re.findall(pat5,text)

                # Level 3 Headings
                level3 = len(l7)

                # Level 2 Headings
                level2 = len(l6) - level3

                # Information noise score
                words = []
                text2 = re.sub(clean, '', text)
                text3 = re.sub(clean2, '', text2)
                text4 = re.sub(clean3, '', text3)
                words += tokeniz.tokenize(text4)

                cfwords = []
                for im in words:
                    cfwords.append(im.lower())
                nonstopwords = []
                for w in cfwords: 
                    if w not in stopw: 
                        nonstopwords.append(w)
                inf_noise_score = 1- (len(nonstopwords)/(art_length))
                
                array1 = np.zeros(num_features)
                if(num_features == 11):
                    # this one is only for content features
                    array1[0] = art_length
                    array1[1] = num_ref
                    array1[2] = cate_links
                    array1[3] = oth_links
                    array1[4] = num_cite_temp
                    array1[5] = num_non_cit_temp
                    array1[6] = num_images
                    array1[7] = inf_tmp
                    array1[8] = level2
                    array1[9] = level3
                    array1[10] = inf_noise_score
                else:
                    # this is for  both content and meta features
                    array1[0] = timestamp2
                    array1[1] = userid
                    array1[2] = size
                    array1[3] = minor2
                    array1[4] = revert
                    array1[5] = comm_len
                    array1[6] = num_ref
                    array1[7] = cate_links
                    array1[8] = oth_links
                    array1[9] = num_cite_temp
                    array1[10] = num_non_cit_temp
                    array1[11] = num_images
                    array1[12] = inf_tmp
                    array1[13] = level2
                    array1[14] = level3
                    array1[15] = inf_noise_score
                
                array2[j] = array1
    
            if(flag1 == 1):
                count2 += 1
            if(flag2 == 0 and flag1 == 0):
                array3[count1] = array2
                count1 += 1
                avail_page_labels.append(page_labels[i])
                avail_pages.append(i)
            else:
                continue
            
    return array3, count1, avail_page_labels, avail_pages            
                

# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for History based LSTM model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages_SplToken1.csv',
                    help='dataset path')
parser.add_argument('--files_path', type=str, nargs='?', default='wiki_files/',
                    help='wikipedia pages path')
parser.add_argument('--num_revisions', type=int, nargs='?', default=10,
                    help='Number of revisions for each page')
parser.add_argument("--only_cont", action='store_true', default = False,
                    help='If true use only content features else use both content and meta features')
parser.add_argument('--num_epoch', type=int, nargs='?', default=20,
                    help='Number of epochs for History based LSTM model')
parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                    help='Training batch size for History based LSTM model')
args = parser.parse_args()


# this csv file has the class of each page
df = pd.read_csv(args.dataset_path)

pages_list = []
page_labels = []
for i in range(len(df)):
    pages_list.append(df['Name'][i] + '0.pkl')
    page_labels.append(df['Label'][i])


if(args.only_cont):
    num_features = 11
else:
    num_features = 16

# Extract the features from all pages
features_array, c, labels, pagenames = extract_features(args.num_revisions, pages_list, 
                                                        page_labels, num_features, args.files_path)


# Shuffle the Data
shuffle = []
for i in range(len(labels)):
    shuffle.append((features_array[i],labels[i]))
random.shuffle(shuffle)

X = []
labels = []
for i in range(len(shuffle)):
    X.append(shuffle[i][0])
    labels.append(shuffle[i][1])
X = np.array(X)


# Transfrom Labels of the page to numerical form
Y = np.array(labels)
le = preprocessing.LabelEncoder()
le.fit(Y)
y = le.transform(Y)


all_features = []
for i in X:
    for j in i:
        all_features.append(j)


# Normalize the Features Extracted
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(all_features)
x = []
for i in range(len(X)):
    x.append(scaler.transform(X[i]))
x = np.array(x)


# Split the data into train, val and text data
x_, x_test, y_, y_test = train_test_split(x, y, test_size = 0.1, random_state = 4)
x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size = 0.1, random_state = 4)


# Conver labels to categorical form
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# Train the Bi-LSTM model
model = Sequential()
model.add(LSTM((100),batch_input_shape = (None,args.num_revisions,num_features),return_sequences=False))
model.add(Dense(6,activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch=args.num_epoch, batch_size=args.batch_size, verbose=2,
          validation_data=(x_val, y_val))


# Predict and Evaluate the model using Test data
pred = model.predict(x_test,verbose = 0)
pred_labels = []
for i in range(len(pred)):
    t2 = list(pred[i])
    pred_labels.append(t2.index(max(t2)))

num_matches = 0
for i in range(len(pred_labels)):
    if(pred_labels[i] == y_test[i]):
        num_matches +=1

print("Accuracy obtained using History based LSTM model is : ",num_matches*100/len(pred_labels))
print("Confusion Matrix of the results obtained using History based LSTM model is :")
confusion_matrix(y_test, pred_labels)

