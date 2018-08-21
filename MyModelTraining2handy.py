# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:14:36 2018

@author: omdiv
"""



#without training embedding 
#Mean Squared Error = 0.0309009
#accuracy = 0.857043719639
#precision_score = 0.778106508876
#recall_score = 0.568443804035
#f1_score = 0.65695253955



from __future__ import print_function
from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn import metrics
import json
import pickle
from collections import Counter
np.random.seed(42)
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional
from keras.layers import LSTM, SimpleRNN, GRU,Flatten,RepeatVector,Permute,Conv1D,GlobalMaxPooling1D
import keras.callbacks
import re
import nltk
from tweet_utils import *

PAD = "<pad>"  
UNK = "<unk>"  
nltk_tokeniser = nltk.tokenize.TweetTokenizer()

EmbeddingSize = 100


def WordEmbeddingLoader(fp, embedding_size):
    embedding = []
    vocab = []
    linenumber=0
    with open(fp, 'r', encoding='UTF-8') as f:
        for each_line in f:
          
            linenumber+=1            
            row = each_line.split(' ')
 
            if len(row) == 2:
                continue
            vocab.append(row[0])
            if len(row[1:]) != embedding_size:
                print (row[0])
                print (len(row[1:]))
            embedding.append(np.asarray(row[1:], dtype='float32'))
   
    word2id = dict(zip(vocab, range(2, len(vocab)+2)))
    word2id[PAD] = 0
    word2id[UNK] = 1

    extra_embedding = [np.zeros(embedding_size), np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)
    return word2id, embedding,vocab

def data_reader(fps, word2id=None, y_len=1, use_target_description=False, use_image=False, delete_irregularities=False):
    ids = []
    post_texts = []
    post_text_lens = []
    truth_means = []
    truth_classes = []
    id2truth_class = {}
    id2truth_mean = {}
    target_descriptions = []
    target_description_lens = []
    image_features = []
    num = 0
    for fp in fps:
        if use_image:
            with open(os.path.join(fp, "id2imageidx.json"), "r") as fin:
                id2imageidx = json.load(fin)
            
            all_image_features = pickle.load(os.path.join(fp, "image_features.hkl"))
        if y_len:
            with open(os.path.join(fp, 'truth.jsonl'), 'rb') as fin:
                for each_line in fin:
                    each_item = json.loads(each_line.decode('utf-8'))
                    if delete_irregularities:
                        if each_item["truthClass"] == "clickbait" and float(each_item["truthMean"]) < 0.5 or each_item["truthClass"] != "clickbait" and float(each_item["truthMean"]) > 0.5:
                            continue
                    if y_len == 4:
                        each_label = [0, 0, 0, 0]
                        
                        for each_key, each_value in Counter(each_item["truthJudgments"]).items():
                            each_label[int(each_key//0.3)] = float(each_value)/5
                        id2truth_class[each_item["id"]] = each_label
                        if each_item["truthClass"] != "clickbait":
                            assert each_label[0]+each_label[1] > each_label[2]+each_label[3]
                        else:
                            assert each_label[0]+each_label[1] < each_label[2]+each_label[3]
                    if y_len == 2:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1, 0]
                        else:
                            id2truth_class[each_item["id"]] = [0, 1]
                    if y_len == 1:
                        if each_item["truthClass"] == "clickbait":
                            id2truth_class[each_item["id"]] = [1]
                        else:
                            id2truth_class[each_item["id"]] = [0]
                    id2truth_mean[each_item["id"]] = [float(each_item["truthMean"])]
        
        with open(os.path.join(fp, 'instances.jsonl'), 'rb') as fin:
            for each_line in fin:
                each_item = json.loads(each_line.decode('utf-8'))
                if each_item["id"] not in id2truth_class and y_len:
                    num += 1
                    continue
                ids.append(each_item["id"])
                each_post_text = " ".join(each_item["postText"])
                each_target_description = each_item["targetTitle"]
                if y_len:
                    truth_means.append(id2truth_mean[each_item["id"]])
                    truth_classes.append(id2truth_class[each_item["id"]])
                if word2id:
                    if (each_post_text+" ").isspace():
                        #the id of <unk>
                        post_texts.append([0])
                        post_text_lens.append(1)
                    else:
                        each_post_tokens = tokeniser(each_post_text)
                        post_texts.append([word2id.get(each_token, 1) for each_token in each_post_tokens])
                        post_text_lens.append(len(each_post_tokens))
                else:
                    post_texts.append([each_post_text])
                if use_target_description:
                    if word2id:
                        if (each_target_description+" ").isspace():
                            target_descriptions.append([0])
                            target_description_lens.append(1)
                        else:
                            each_target_description_tokens = tokeniser(each_target_description)
                            target_descriptions.append([word2id.get(each_token, 1) for each_token in each_target_description_tokens])
                            target_description_lens.append(len(each_target_description_tokens))
                    else:
                        target_descriptions.append([each_target_description])
                else:
                    target_descriptions.append([])
                    target_description_lens.append(0)
                if use_image:
                    image_features.append(all_image_features[id2imageidx[each_item["id"]]].flatten())
                else:
                    image_features.append([])
    print ("Deleted number of items: " + str(num))
    return ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features



def Sequence_pader(sequences, maxlen):
    if maxlen <= 0:
        return sequences
    shape = (len(sequences), maxlen)
    padded_sequences = np.full(shape, 0)
    for i, each_sequence in enumerate(sequences):
        if len(each_sequence) > maxlen:
            padded_sequences[i] = each_sequence[:maxlen]
        else:
            padded_sequences[i, :len(each_sequence)] = each_sequence
    return padded_sequences



def tweet_tokenizer(text):
    return simpleTokenize(squeezeWhitespace(text))

def tokeniser(text, with_process=True):
    if with_process:
        return nltk_tokeniser.tokenize(tweet_processor(text).lower())
    else:
        # return nltk_tokeniser.tokenize(text)
        return tweet_tokenizer(text.lower())




def tweet_processor(text):
    FLAGS = re.MULTILINE | re.DOTALL
    
    def megasplit(pattern, string):
        splits = list((m.start(), m.end()) for m in re.finditer(pattern, string))
        starts = [0] + [i[1] for i in splits]
        ends = [i[0] for i in splits] + [len(string)]
        return [string[start:end] for start, end in zip(starts, ends)]
    
    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        #print(hashtag_body)
        
        #result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        result = " ".join(["<hashtag>"] + megasplit(r"(?=[A-Z])", hashtag_body))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text





np.random.seed(81)
word2id, embedding_matrix,vocab = WordEmbeddingLoader(fp=os.path.join('data', "glove.6B."+str(EmbeddingSize)+"d.txt"), embedding_size=EmbeddingSize)
with open(os.path.join('data', 'word2id.json'), 'w') as fout:
    json.dump(word2id, fp=fout)
    

ids, post_texts, truth_classes, post_text_lens, truth_means, target_descriptions, target_description_lens, image_features = data_reader(word2id=word2id, fps=[os.path.join('data', 'clickbait17-validation'), os.path.join('data', 'clickbait17-train-170331')], y_len=4, use_target_description=False, use_image=False)
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
target_descriptions = Sequence_pader(target_descriptions, max_target_description_len)




tetids, tepost_texts, tetruth_classes, tepost_text_lens, tetruth_means, tetarget_descriptions, tetarget_description_lens, teimage_features = data_reader(word2id=word2id, fps=[os.path.join('data', 'clickbait17-test')], y_len=4, use_target_description=False, use_image=False)    
tepost_texts = np.array(tepost_texts)
tetruth_classes = np.array(tetruth_classes)
tepost_text_lens = [each_len if each_len <= max_post_text_len else max_post_text_len for each_len in tepost_text_lens]
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


model.add(Embedding(input_dim = max_features,
                    output_dim = embedding_dims,
                    weights = [embedding_matrix],
                    input_length = maxlen
                    ,trainable = False))
model.add(Dropout(dropout_embedding))

model.add(Bidirectional(GRU(512, dropout_W=0.2, dropout_U=0.5)))  
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop')

batch_size = 64

earlystop_cb = keras.callbacks.EarlyStopping(monitor='mse', patience=7, verbose=1, mode='auto')

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



mse = metrics.mean_squared_error(tetruth_means,petruth_means)
print('Mean Squared Error = '+str(mse))

accuracy = metrics.accuracy_score(tetruthClass,petruthClass)
print('accuracy = '+str(accuracy))

precision = metrics.precision_score(tetruthClass,petruthClass)
print('precision_score = '+str(precision))

recall = metrics.recall_score(tetruthClass,petruthClass)
print('recall_score = '+str(recall))

f1 = metrics.f1_score(tetruthClass,petruthClass)
print('f1_score = '+str(f1))