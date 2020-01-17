import json
import os
import pickle
import re
from collections import Counter

import nltk
import numpy as np

from tweet_utils import simpleTokenize, squeezeWhitespace

PAD = "<pad>"
UNK = "<unk>"
NLTK_TOKENIZER = nltk.tokenize.TweetTokenizer()


def WordEmbeddingLoader(fp, embedding_size):
    embedding = []
    vocab = []
    linenumber = 0
    with open(fp, 'r', encoding='UTF-8') as f:
        for each_line in f:

            linenumber += 1
            row = each_line.split(' ')

            if len(row) == 2:
                continue
            vocab.append(row[0])
            if len(row[1:]) != embedding_size:
                print(row[0])
                print(len(row[1:]))
            embedding.append(np.asarray(row[1:], dtype='float32'))

    word2id = dict(zip(vocab, range(2, len(vocab)+2)))
    word2id[PAD] = 0
    word2id[UNK] = 1

    extra_embedding = [np.zeros(embedding_size),
                       np.random.uniform(-0.1, 0.1, embedding_size)]
    embedding = np.append(extra_embedding, embedding, 0)
    return word2id, embedding, vocab


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

            all_image_features = pickle.load(
                os.path.join(fp, "image_features.hkl"))
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
                            each_label[int(each_key//0.3)
                                       ] = float(each_value)/5
                        id2truth_class[each_item["id"]] = each_label
                        if each_item["truthClass"] != "clickbait":
                            assert each_label[0] + \
                                each_label[1] > each_label[2]+each_label[3]
                        else:
                            assert each_label[0] + \
                                each_label[1] < each_label[2]+each_label[3]
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
                    id2truth_mean[each_item["id"]] = [
                        float(each_item["truthMean"])]

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
                        # the id of <unk>
                        post_texts.append([0])
                        post_text_lens.append(1)
                    else:
                        each_post_tokens = tokeniser(each_post_text)
                        post_texts.append([word2id.get(each_token, 1)
                                           for each_token in each_post_tokens])
                        post_text_lens.append(len(each_post_tokens))
                else:
                    post_texts.append([each_post_text])
                if use_target_description:
                    if word2id:
                        if (each_target_description+" ").isspace():
                            target_descriptions.append([0])
                            target_description_lens.append(1)
                        else:
                            each_target_description_tokens = tokeniser(
                                each_target_description)
                            target_descriptions.append(
                                [word2id.get(each_token, 1) for each_token in each_target_description_tokens])
                            target_description_lens.append(
                                len(each_target_description_tokens))
                    else:
                        target_descriptions.append([each_target_description])
                else:
                    target_descriptions.append([])
                    target_description_lens.append(0)
                if use_image:
                    image_features.append(
                        all_image_features[id2imageidx[each_item["id"]]].flatten())
                else:
                    image_features.append([])
    print("Deleted number of items: " + str(num))
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
        return NLTK_TOKENIZER.tokenize(tweet_processor(text).lower())
    else:
        # return NLTK_TOKENIZER.tokenize(text)
        return tweet_tokenizer(text.lower())


def tweet_processor(text):
    FLAGS = re.MULTILINE | re.DOTALL

    def megasplit(pattern, string):
        splits = list((m.start(), m.end())
                      for m in re.finditer(pattern, string))
        starts = [0] + [i[1] for i in splits]
        ends = [i[0] for i in splits] + [len(string)]
        return [string[start:end] for start, end in zip(starts, ends)]

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        # print(hashtag_body)

        #result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        result = " ".join(
            ["<hashtag>"] + megasplit(r"(?=[A-Z])", hashtag_body))
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
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes,
                                                  nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes,
                                            nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text
