# -*- coding: utf-8 -*-
# @Author: feidong
# @Date:   2017-01-04 15:51:15
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-01-12 21:38:17

import os
import codecs
from model import *
import numpy as np


def load_vocab(vocabfile):
    vocab = {}
    with codecs.open(vocabfile, 'r', encoding='utf-8') as f:
        # f.next()
        for line in f.readlines():
            line = line.strip()
            key, value = line.split()
            if key not in vocab:
                vocab[key] = int(value)
    idx2instance = {v: k for k, v in vocab.iteritems()}
    # print idx2instance[299]
    return vocab, idx2instance


def save_w2v(embeddings, idx2instance, vecfile):
    embedL = embeddings.tolist()
    with codecs.open(vecfile, 'w', encoding='utf-8') as f:
        for i in xrange(len(embedL)):
            word = idx2instance[i]
            value = [str(v) for v in embedL[i]]
            w2v = [word] + value
            w2v_format = " ".join(w2v)
            f.write("%s\n" % w2v_format)


def save_weights(wList, outpath):
    assert len(wList) == 4, 'length should be 4, w1, b1, w2, b2'
    W1, b1, W2, b2 = wList
    with open(outpath, 'w') as f:
        #write w1
        f.write("W1: %s %s\n" % (W1.shape[0], W1.shape[1]))
        wL = W1.tolist()
        for i in xrange(len(wL)):
            valueL = [str(v) for v in wL[i]]
            value_format = " ".join(valueL)
            f.write("%s\n" % value_format)

        # write b1
        f.write("b1: %s\n" % (b1.shape[0]))
        bL = b1.tolist()
        valueL = [str(v) for v in bL]
        value_format = " ".join(valueL)
        f.write("%s\n" % value_format)

        #write w2
        f.write("W2: %s %s\n" % (W2.shape[0], W2.shape[1]))
        wL = W2.tolist()
        for i in xrange(len(wL)):
            valueL = [str(v) for v in wL[i]]
            value_format = " ".join(valueL)
            f.write("%s\n" % value_format)

        # write b2
        f.write("b2: %s\n" % (b2.shape[0]))
        bL = b2.tolist()
        valueL = [str(v) for v in bL]
        value_format = " ".join(valueL)
        f.write("%s\n" % value_format)   


def save_W(wM, outpath):
    wL = wM.tolist()
    with open(outpath, 'w') as f:
        for i in xrange(len(wL)):
            valueL = [str(v) for v in wL[i]]
            value_format = " ".join(valueL)
            f.write("%s\n" % value_format)


def save_bias(bV, outpath):
    bL = bV.tolist()
    with open(outpath, 'w') as f:
        valueL = [str(v) for v in bL]
        value_format = " ".join(valueL)
        f.write("%s\n" % value_format)


def extractor_embeddings(model, uni_vocabfile, bi_vocabfile, uni_vecfile, bi_vecfile):
    univocab, uni_idx2indices = load_vocab(uni_vocabfile)
    bivocab, bi_idx2indices = load_vocab(bi_vocabfile)

    univocab_size = len(univocab)
    bivocab_size = len(bivocab)
    print univocab_size, bivocab_size

    if 0 not in uni_idx2indices:
        uniL = range(1, univocab_size)
    else:
        uniL = range(0, univocab_size-1)
    if 0 not in bi_idx2indices:
        biL = range(1, bivocab_size)
    else:
        biL = range(0, bivocab_size-1)
    # print uniL, biL
    uniMat = np.asarray(uniL).reshape(len(uniL), 1)
    uni_data = np.repeat(uniMat, 5, axis=1)

    biMat = np.asarray(biL).reshape(len(biL), 1)
    bi_data = np.repeat(biMat, 5, axis=1)

    uni_embed_name = 'x1'
    bi_embed_name = 'x2'

    # print uni_data[-10:], bi_data[-10:]
    uni_embeddings = get_embed_vec(model, 'x1', [uni_data, bi_data])
    bi_embeddings = get_embed_vec(model, 'x2', [bi_data, bi_data])

    print uni_embeddings.shape, bi_embeddings.shape
    # print uni_embeddings[0:1], bi_embeddings[0]
    # print type(uni_embeddings)
    save_w2v(uni_embeddings, uni_idx2indices, uni_vecfile)
    save_w2v(bi_embeddings, bi_idx2indices, bi_vecfile)
    return uni_embeddings, bi_embeddings


def extractor_weights(model, layername='z'):
    Wb = get_layer_weights(model, layername)
    # assert len(Wb) == 2
    if len(Wb) == 1:    # for layer without bias
        print Wb[0].shape
        return Wb[0], None
    elif len(Wb) == 2:  # with bias
        W, b = Wb
        print W.shape, b.shape
        return W, b
    else:
        return None, None