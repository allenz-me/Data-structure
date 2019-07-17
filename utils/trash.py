import time
import gc
import sys
import os
import csv
import itertools
import random
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import sparse
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# import lightgbm as lgb

import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors

import keras
from keras.layers import *
from keras.models import Sequential, Model



s = set()
f = open("/home/kesci/work/helpermodel/LineSentence.cor", "w")



# In [1]:

def load_sentence():
    # 加载并保存 title
    f = open('title.txt', 'w')
    with open('/home/kesci/input/bytedance/first-round/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(csv_reader):
            title = row[3]
            f.write(f"{title}\n")
    f.close()

LineSen = word2vec.LineSentence('title.txt')
model = word2vec.Word2Vec(sentences=LineSen,
		 min_count=2, size=100, workers=5)
wv = model.wv

# In [2]:

def load_train():
    # 加载前一千万
    f = open('title.txt')
    all_vectors = []
    for i in tqdm(range(20 ** 5)):
        temp = np.zeros(100)
        count = 0
        words = f.readline().split()
        for w in words:
            if w in wv.vocab:
                count += 1
                temp += wv.word_vec(w)
        temp = temp / count if count else temp
        all_vectors.append(temp)
    return np.vstack(all_vectors)

def load_test():
    f = open('test_title.txt')
    test_vectors = []
    for i in trange(500_0000):
        temp = np.zeros(100)
        count = 0
        words = f.readline().split()
        for w in words:
            if w in wv.vocab:
                count += 1
                temp += wv.word_vec(w)
        temp = temp / count if count else temp
        test_vectors.append(temp)
    return np.vstack(test_vectors)

# In [3]:

class roc_callback(keras.callbacks.Callback):
    def __init__(self,xtrain, ytrain, xval=None, yval=None):
        self.x = xtrain
        self.y = ytrain
        
        assert xval and yval
        if xval and yval:
            self.xval = xval
            self.yval = yval
        
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)         
        print(f'epoch {epoch}, roc-auc: {roc:.5f}', end='\t' )
        
        if hasattr(self, 'xval'):
            y_pred = self.model.predict(self.xval)
            roc_val = roc_auc_score(self.yval, y_pred)
            print(f'roc-auc: {roc_val:.5f}', end='')
        print()
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

model = Sequential()
model.add(Dense(input_dim=100, units=16, activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', 
	optimizer='rmsprop', metrics=['accuracy'])