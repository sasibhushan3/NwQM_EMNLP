
''' Code to run doc2vec model to generate doc2vec vector representation
    of a wikipedia article/page and run Logistic Regression (or) Random
    Forests to classify these vectors in the 6 Wikipedia Quality Classes
                    (FA, GA, B, C, Start, Stub)
'''
import numpy as np
import pandas as pd
import gensim
import os
import collections
import random
import sys
import csv
import argparse
from math import floor, ceil
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# Read the csv file
def readCsv(fname):
    with open(fname, encoding="utf8") as csvFile:
        reader1 = csv.reader(csvFile)
        list1 = list(reader1)   
    csvFile.close()
    return list1


# Pre-Process the data
def readCorpus(list1):
    sum = 0
    for i in range(len(list1)):
        if(i>0):
            sum = sum +len(list1[i][1])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(list1[i][2])[0:],list1[i][1])
    

# Prevent OverflowError
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for doc2vec model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages_SplToken1.csv',
                    help='dataset path')
parser.add_argument('--num_epoch', type=int, nargs='?', default=55,
                    help='Number of epochs for doc2vec model')
args = parser.parse_args()


# Pre-Process Data
read_data = (readCsv(args.dataset_path))
processed_data = list(readCorpus(read_data))
total_num_obs = len(processed_data)


# Divide the Data into train and test
train_corpus = processed_data[0:20000]
test_corpus = processed_data[24000:30000]


# Train the doc2vec model using train corpus
model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=2, iter=args.num_epoch)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)


''' train_corpus and test_corpus contains both the processed data and label for each page.
    So divide them into (train_targets, train_regressors) and (test_targets, test_regressors)
    respectively.
'''
train_targets, train_regressors = zip(*[(doc.words, doc.tags) for doc in train_corpus])
test_targets, test_regressors = zip(*[(doc.words, doc.tags) for doc in test_corpus])


# Generate the doc2vec vector representation for train data.
X = []
for i in range(len(train_targets)):
    print(i)
    X.append(model.infer_vector(train_targets[i]))
train_x = np.asarray(X)


# Convert the train labels into integers.
Y = np.asarray(train_regressors)
le = preprocessing.LabelEncoder()
le.fit(Y)
train_y = le.transform(Y)


# Generate the doc2vec vector representation for test data.
test_list = []
for i in range(len(test_targets)):
    test_list.append(model.infer_vector(test_targets[i]))
test_x = np.asarray(test_list)


# Convert the test labels into integers.
test_Y = np.asarray(test_regressors)
test_y = le.transform(test_Y)


# Run Logistic Regression and predict the labels for test data
logreg = linear_model.LogisticRegression()
logreg.fit(train_x, train_y)
preds = logreg.predict(test_x)


# Run Random Forest Classifier and predict the labels for test data
randomfor=  RandomForestClassifier(n_estimators=100,max_depth = 5,random_state = 0)
randomfor.fit(train_x, train_y)
preds2 = randomfor.predict(test_x)


# Results Obtained using Logistic Regression
print("Accuracy obtained using logistic regression is : ",sum(preds == test_y)*100/ len(test_y))
print("Confusion Matrix of the results obtained using logistic regression is :")
print(confusion_matrix(test_y, preds))


# Results Obtained using Random Forests
print("Accuracy obtained using random forests is : ",sum(preds2 == test_y)*100/ len(test_y))
print("Confusion Matrix of the results obtained using random forests is :")
print(confusion_matrix(test_y, preds2))
