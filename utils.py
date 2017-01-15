# -*- coding: utf-8 -*-
# @Author: feidong
# @Date:   2017-01-02 14:48:55
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-01-14 17:08:11

import os
import sys
import codecs
import logging
from alphabet import Alphabet
from loadEmb import W2VEmbReader
from collections import defaultdict
import numpy as np

NULL = '-null-'
padding_word = '-padding-'
unknown_word = '-unknown-'

# logger = logging.getLogger(__name__)


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def data_reader(path):
    charL, labelL = [], []
    sentencesList = []
    labelList = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        # f.next()
        for line in f.readlines():
            line = line.strip()
            if line:
                char = line.split()[0]
                if not len(line.split()) == 2:
                    print line
                label = line.split()[1]
                charL.append(char)
                labelL.append(label)
            else:
                sentencesList.append(charL)
                labelList.append(labelL)
                charL, labelL = [], []
        if charL and labelL:
            sentencesList.append(charL)
            labelList.append(labelL)
        # print labelList[-1]
    return sentencesList, labelList


# def get_unigram(path, vocab):
#     sentencesList, _ = data_reader(path)
#     vocab.get_index(padding_word)
#     for sentence in sentencesList:
#         for w in sentence:
#             voab.get_index(w)
#     return vocab


def generate_context(sentence, context_size=2):
    offset = context_size
    window_size = 2*offset + 1

    context_uni_sent = []
    context_bi_sent = []
    unigrams, bigrams = [], []

    for i in xrange(len(sentence)):
        if i == len(sentence)-1:
            bigram = sentence[i] + NULL
        else:
            bigram = sentence[i] + sentence[i+1]
        unigram = sentence[i]
        unigrams.append(unigram)
        bigrams.append(bigram)
    context_uni_sent = windowlize(unigrams, context_size)
    context_bi_sent = windowlize(bigrams, context_size)

    # print_examples(sentence)

    # print_examples(context_uni_sent)

    # print_examples(context_bi_sent)
    # sys.exit(0)
    assert len(context_bi_sent) == len(sentence)
    return context_uni_sent, context_bi_sent


def print_examples(L):
    if isinstance(L[0], list):
        for l in L:
            for s in l:
                print s
    else:
        for s in L:
            print s
    print "\n"


def windowlize(L, context_size=2):
    contextL = []
    for i in xrange(len(L)):
        context_words = []
        for idy in range(i-context_size, i+context_size+1):
            if (idy < 0) or (idy > len(L)-1):
                context_words.append(padding_word)
            else:
                context_words.append(L[idy])
        contextL.append(context_words)
    return contextL


def get_sent_length(sents):
    slen = []
    for s in sents:
        slen.append(len(s))
    return slen


def generate_context_data(path, context_size):

    sentencesList, labelList = data_reader(path)
    sentlengthList = get_sent_length(labelList)
    uni_contexts = []
    bi_contexts = []

    for sentence in sentencesList:
        uni_sentence, bi_sentence = generate_context(sentence, context_size)
        uni_contexts.append(uni_sentence)
        bi_contexts.append(bi_sentence)

    return sentencesList, labelList, sentlengthList, uni_contexts, bi_contexts


def save_vocab(word_vocab, vocabfile, ngram=1):
    with codecs.open(vocabfile, 'w', encoding='utf-8') as f:
        for w, idx in word_vocab.iteritems():
            # print w, idx
            f.write("%s %s\n" %(w, idx))
        # elif ngram == 2:
        #     for w, idx in word_vocab.iteritems():
        #         concat_w = "_".join(w)
        #         f.write("%s %s\n" % (concat_w, idx))


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([word_alphabet.size(), embedd_dim], dtype=theano.config.floatX)
    embedd_table[word_alphabet.default_index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    oov_num = 0
    for word, index in word_alphabet.iteritems():
        ww = word.lower() if caseless else word
        # show oov ratio
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[index, :] = embedd
    oov_ratio = float(oov_num)/(word_alphabet.size()-1)
    logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table


def creat_data(trainpath=None, devpath=None, testpath=None, context_size=2, uni_vocabfile=None, bi_vocabfile=None, embed_dim=50, \
             uni_embpath=None, bi_embpath=None, load_biEmb=True):

    def get_indexes(sentences, vocab):
        total_count = 0
        oov_count = 0
        index_sentences = []
        i = 0
        for sentence in sentences:
            if (i % 10000 == 0):
                print "Current sentence num = %s" % i
            i += 1
            s_indexes = []
            for context in sentence:
                w_indexes = []
                for w in context:
                    if w not in vocab:
                        idx = vocab[unknown_word]
                        oov_count += 1
                    else:
                        idx = vocab[w]
                    total_count += 1
                    w_indexes.append(idx)
                s_indexes.append(w_indexes)
            index_sentences.append(s_indexes)
        logger.info("OOV words count = %s, total words count = %s, OOV ratio = %s" % \
            (oov_count, total_count, str(float(oov_count)/total_count)))
        return index_sentences

    def get_label_index(labelList):
        label_indexes = []
        label_vocab = {}
        # idx = 0
        next_idx = 0
        for sentlabels in labelList:
            label_index = []
            for label in sentlabels:
                if label not in label_vocab:
                    label_vocab[label] = next_idx
                    label_index.append(next_idx)
                    # idx = next_idx
                    next_idx += 1
                else:
                    idx = label_vocab[label]
                    label_index.append(idx)
            label_indexes.append(label_index)
        print label_vocab
        return label_indexes, label_vocab

    def generate_matrix(indexes):
        flatten_indexes = [c for s in indexes for c in s]
        return np.asarray(flatten_indexes)

    def create_univocab(sentences, uni_vocab, next_idx):
        for s in sentences:
            for w in s:
                # uni_vocab.get_index(w)
                if w not in uni_vocab:
                    uni_vocab[w] = next_idx
                    next_idx += 1
        return uni_vocab, next_idx

    def create_bivocab(sentences, bi_vocab, next_idx):
        i = 0
        cnt = 0
        for s in sentences:
            if (cnt % 1000) == 0:
                print "Current sent num = %s" % cnt
            cnt += 1
            for i in xrange(len(s)):
                if i == len(s)-1:
                    bigram = s[i] + NULL
                else:
                    bigram = s[i] + s[i+1]
                # bi_vocab.get_index(bigram)
                if bigram not in bi_vocab:
                    bi_vocab[bigram] = next_idx
                    next_idx += 1
        return bi_vocab, next_idx
    logger = get_logger("Creat data ...")
    train_sentsList, train_labelList,  train_sentlenL, train_uni_contexts, train_bi_contexts = \
        generate_context_data(trainpath, context_size)
    logger.info("Total number of sentences in training data = %s" % len(train_sentsList))

    if devpath:
        dev_sentsList, dev_labelList,  dev_sentlenL, dev_uni_contexts, dev_bi_contexts = \
            generate_context_data(devpath, context_size)
        logger.info("Total number of sentences in dev data = %s" % len(dev_sentsList))

    test_sentsList, test_labelList, test_sentlenL, test_uni_contexts, test_bi_contexts = \
        generate_context_data(testpath, context_size)
    logger.info("Total number of sentences in test data = %s" % len(test_sentsList))
    logger.info('Creating vocab...')
    # uni_vocab = Alphabet('unigrams')
    # uni_vocab.get_index(padding_word)
    # uni_vocab.get_index(unknown_word)
    uni_vocab = {}
    uni_vocab[padding_word] = 0
    uni_vocab[unknown_word] = 1
    next_idx = 2

    uni_vocab, next_idx = create_univocab(train_sentsList, uni_vocab, next_idx)
    if devpath:
        uni_vocab, next_idx = create_univocab(dev_sentsList, uni_vocab, next_idx)
    uni_vocab, next_idx = create_univocab(test_sentsList, uni_vocab, next_idx)
    # uni_vocab.close()
    logger.info("unigrams alphabet size = %s" % (len(uni_vocab)-1))    
    # bi_vocab = Alphabet('bigrams')
    # bi_vocab.get_index(padding_word)
    # bi_vocab.get_index(unknown_word)
    bi_vocab = {}
    bi_vocab[padding_word] = 0
    bi_vocab[unknown_word] = 1
    next_idx = 2
    bi_vocab, next_idx = create_bivocab(train_sentsList, bi_vocab, next_idx)

    if devpath:
        bi_vocab, next_idx = create_bivocab(dev_sentsList, bi_vocab, next_idx)    
    bi_vocab, next_idx = create_bivocab(test_sentsList, bi_vocab, next_idx)
    # bi_vocab.close()

    logger.info("bigrams alphabet size = %s" % (len(bi_vocab)-1))

    # train_uni_indexes = get_uni_indexes(train_uni_contexts)
    # test_uni_indexes = get_uni_indexes(test_uni_contexts)
    # train_bi_indexes = get_bi_indexes(train_bi_contexts)
    # test_bi_indexes = get_bi_indexes(test_bi_contexts)
    train_uni_indexes = get_indexes(train_uni_contexts, uni_vocab)
    test_uni_indexes = get_indexes(test_uni_contexts, uni_vocab)
    train_bi_indexes = get_indexes(train_bi_contexts, bi_vocab)
    test_bi_indexes = get_indexes(test_bi_contexts, bi_vocab)


    train_label_indexes, train_label_vocab = get_label_index(train_labelList)
    test_label_indexes, test_label_vocab = get_label_index(test_labelList)
    # assert len(train_label_vocab) == len(test_label_vocab)

    uni_train = generate_matrix(train_uni_indexes)
    bi_train = generate_matrix(train_bi_indexes)
    label_train = generate_matrix(train_label_indexes)

    uni_test = generate_matrix(test_uni_indexes)
    bi_test = generate_matrix(test_bi_indexes)
    label_test = generate_matrix(test_label_indexes)

    if devpath:
        dev_uni_indexes = get_indexes(dev_uni_contexts, uni_vocab)
        dev_bi_indexes = get_indexes(dev_bi_contexts, bi_vocab)
        dev_label_indexes, dev_label_vocab = get_label_index(dev_labelList)

        uni_dev = generate_matrix(dev_uni_indexes)
        bi_dev = generate_matrix(dev_bi_indexes)
        label_dev = generate_matrix(dev_label_indexes)

    label_vocab = train_label_vocab
    logger.info("label alphabet size = %s" % len(label_vocab))

    save_vocab(uni_vocab, uni_vocabfile, ngram=1)
    save_vocab(bi_vocab, bi_vocabfile, ngram=2)

    logger.info("Loading pretrained word unigram embedding ...")
    uni_embReader = W2VEmbReader(uni_embpath, uni_vocab, embed_dim)

    uni_embed_table = np.empty((len(uni_vocab), embed_dim))
    uni_embed_table = uni_embReader.get_emb_matrix_given_vocab(uni_vocab, uni_embed_table)

    if load_biEmb:
        logger.info("Loading pretrained word bigram embedding ...")
        bi_embReader = W2VEmbReader(bi_embpath, bi_vocab, embed_dim)
        bi_embed_table = np.empty((len(bi_vocab), embed_dim))

        bi_embed_table = bi_embReader.get_emb_matrix_given_vocab(bi_vocab, bi_embed_table)
    else:
        bi_embed_table = None

    if not devpath:
        return uni_train, bi_train, None, None, uni_test, bi_test, label_train, None, label_test, \
            uni_vocab, bi_vocab, label_vocab, [uni_embed_table], [bi_embed_table], train_sentlenL, None, test_sentlenL

    else:
        return uni_train, bi_train, uni_dev, bi_dev,  uni_test, bi_test, label_train, label_dev, label_test, \
            uni_vocab, bi_vocab, label_vocab, [uni_embed_table], [bi_embed_table], train_sentlenL, dev_sentlenL, test_sentlenL 
