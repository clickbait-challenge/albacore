# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:14:36 2018

@author: omdiv
"""


# without training embedding
# Mean Squared Error = 0.0309009
#accuracy = 0.857043719639
#precision_score = 0.778106508876
#recall_score = 0.568443804035
#f1_score = 0.65695253955


from __future__ import print_function
from keras.layers import LSTM, SimpleRNN, GRU, Flatten, RepeatVector, Permute, Conv1D, GlobalMaxPooling1D
from tweet_utils import *
from utils import *
import nltk
import re
import keras.callbacks
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional
from keras.models import Sequential
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn import metrics
import json
import pickle
from collections import Counter

np.random.seed(42)


EmbeddingSize = 100


np.random.seed(81)
word2id, embedding_matrix, vocab = WordEmbeddingLoader(fp=os.path.join(
    'data', "glove.6B."+str(EmbeddingSize)+"d.txt"), embedding_size=EmbeddingSize)
with open(os.path.join('data', 'word2id.json'), 'w') as fout:
    json.dump(word2id, fp=fout)


ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features = data_reader(
    word2id=word2id, fps=[os.path.join('data', 'clickbait17-validation'), os.path.join('data', 'clickbait17-train-170331')], y_len=4, use_target_description=False, use_image=False)
post_texts = np.array(post_texts)
truth_classes = np.array(truth_classes)
post_text_lens = np.array(post_text_lens)
truth_means = np.array(truth_means)
shuffle_indices = np.random.permutation(np.arange(len(post_texts)))
post_texts = post_texts[shuffle_indices]
truth_classes = truth_classes[shuffle_indices]
post_text_lens = post_text_lens[shuffle_indices]
truth_means = truth_means[shuffle_indices]
max_post_text_len = max(post_text_lens)
print(max_post_text_len)

post_texts = Sequence_pader(post_texts, max_post_text_len)

target_descriptions = np.array(target_descriptions)
target_description_lens = np.array(target_description_lens)
target_descriptions = target_descriptions[shuffle_indices]
target_description_lens = target_description_lens[shuffle_indices]
max_target_description_len = max(target_description_lens)
print(max_target_description_len)
target_descriptions = Sequence_pader(
    target_descriptions, max_target_description_len)


tetids, tepost_texts, tetruth_classes, tepost_text_lens, tetruth_means, tetarget_descriptions, tetarget_description_lens, teimage_features = data_reader(
    word2id=word2id, fps=[os.path.join('data', 'clickbait17-test')], y_len=4, use_target_description=False, use_image=False)
tepost_texts = np.array(tepost_texts)
tetruth_classes = np.array(tetruth_classes)
tepost_text_lens = [each_len if each_len <=
                    max_post_text_len else max_post_text_len for each_len in tepost_text_lens]
tepost_text_lens = np.array(tepost_text_lens)
tetruth_means = np.array(tetruth_means)
tetruth_means = np.ravel(tetruth_means).astype(np.float32)
tepost_texts = Sequence_pader(tepost_texts, max_post_text_len)


max_features = len(word2id.keys())
maxlen = max_post_text_len
embedding_dims = EmbeddingSize
dropout_embedding = 0.2
X_train = post_texts
y_train = truth_means

X_test = tepost_texts
y_test = tetruth_means


# build the keras LSTM model
model = Sequential()


model.add(Embedding(input_dim=max_features,
                    output_dim=embedding_dims,
                    weights=[embedding_matrix],
                    input_length=maxlen, trainable=False))
model.add(Dropout(dropout_embedding))

model.add(Bidirectional(GRU(512, dropout_W=0.2, dropout_U=0.5)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop')

batch_size = 64

earlystop_cb = keras.callbacks.EarlyStopping(
    monitor='mse', patience=7, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20,
          validation_split=0.1, callbacks=[earlystop_cb])

petruth_means = model.predict(X_test)
tetruthClass = []
petruthClass = []


for i in range(len(tetruth_means)):

    if petruth_means[i] > 0.5:
        petruthClass.append(1)
    else:
        petruthClass.append(0)

    if tetruth_means[i] > 0.5:
        tetruthClass.append(1)
    else:
        tetruthClass.append(0)


mse = metrics.mean_squared_error(tetruth_means, petruth_means)
print('Mean Squared Error = '+str(mse))

accuracy = metrics.accuracy_score(tetruthClass, petruthClass)
print('accuracy = '+str(accuracy))

precision = metrics.precision_score(tetruthClass, petruthClass)
print('precision_score = '+str(precision))

recall = metrics.recall_score(tetruthClass, petruthClass)
print('recall_score = '+str(recall))

f1 = metrics.f1_score(tetruthClass, petruthClass)
print('f1_score = '+str(f1))
