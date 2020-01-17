# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:14:36 2018

@author: omdiv
"""

# BIdirectional + Convolution layer

# Mean Squared Error = 0.0327794
#accuracy = 0.844899375434
#precision_score = 0.767895878525
#recall_score = 0.510086455331
#f1_score = 0.612987012987


from __future__ import print_function

import json
import os
import pickle
import re
from collections import Counter

import keras.callbacks
import nltk
import numpy as np
from gensim.models import Word2Vec
from keras.layers import (GRU, LSTM, Activation, Bidirectional, Conv1D, Dense,
                          Dropout, Embedding, Flatten, GlobalMaxPooling1D,
                          Permute, RepeatVector, SimpleRNN)
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

from tweet_utils import *
from utils import *

np.random.seed(42)


PAD = "<pad>"
UNK = "<unk>"
nltk_tokeniser = nltk.tokenize.TweetTokenizer()


def main(argv=None):

    with open('MyModelTraining2optimise.txt', "w") as myfile:
        myfile.write(
            'statesize,EmbeddingSize,dropout_embedding,dropout_W,dropout_U,mse, accuracy, precision, recall, f1 \n')

    for EmbeddingSize in [300, 200, 100, 50]:

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

        post_texts = Sequence_pader(post_texts, max_post_text_len)

        target_descriptions = np.array(target_descriptions)
        target_description_lens = np.array(target_description_lens)
        target_descriptions = target_descriptions[shuffle_indices]
        target_description_lens = target_description_lens[shuffle_indices]
        max_target_description_len = max(target_description_lens)

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

        X_train = post_texts
        y_train = truth_means

        X_test = tepost_texts
        y_test = tetruth_means

        epochs = 20
        batch_size = 64

        for dropout_embedding in [0.2, 0.5]:
            for dropout_W in [0.2, 0]:
                for dropout_U in [0.2, 0.3, 0.5]:
                    for statesize in [128, 256]:

                        # build the keras LSTM model
                        model = Sequential()

                        # build the keras LSTM model

                        model.add(Embedding(input_dim=max_features,
                                            output_dim=embedding_dims,
                                            weights=[embedding_matrix],
                                            input_length=maxlen, trainable=False))
                        model.add(Dropout(dropout_embedding))

                        # try using a GRU instead, for fun   #
                        model.add(Bidirectional(
                            GRU(statesize, dropout_W=dropout_W, dropout_U=dropout_U)))

                        model.add(Dense(1))
                        model.add(Activation('sigmoid'))

#                        model.summary()

                        # try using different optimizers and different optimizer configs
                        model.compile(loss='mse',
                                      optimizer='rmsprop')

                        print('Train...')
                        earlystop_cb = keras.callbacks.EarlyStopping(
                            monitor='mse', patience=7, verbose=1, mode='auto')

                        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                                  validation_split=0.1, callbacks=[earlystop_cb])
                        # score, acc = model.evaluate(X_test, y_test,
                        #                            batch_size=batch_size)

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

                        mse = metrics.mean_squared_error(
                            tetruth_means, petruth_means)
                        print('Mean Squared Error = '+str(mse))

                        accuracy = metrics.accuracy_score(
                            tetruthClass, petruthClass)
                        print('accuracy = '+str(accuracy))

                        precision = metrics.precision_score(
                            tetruthClass, petruthClass)
                        #print('precision_score = '+str(precision))

                        recall = metrics.recall_score(
                            tetruthClass, petruthClass)
                        #print('recall_score = '+str(recall))

                        f1 = metrics.f1_score(tetruthClass, petruthClass)
                        #print('f1_score = '+str(f1))

                        with open('MyModelTraining2optimise.txt', "a") as myfile:
                            myfile.write(str(statesize)+','+str(EmbeddingSize)+','+str(dropout_embedding)+','+str(dropout_W)+','+str(
                                dropout_U)+','+str(mse)+','+str(accuracy) + ','+str(precision)+','+str(recall)+','+str(f1) + ' \n')
#                        print(str(statesize)+','+str(EmbeddingSize)+','+str(dropout_embedding)+','+str(CNN_filters)+','+str(dropout_W)+','+str(dropout_U)+','+str(mse)+' \n')


if __name__ == "__main__":

    main()
