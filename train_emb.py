# -*- coding: utf-8 -*-
# @Author: feidong
# @Date:   2017-01-02 21:03:05
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-01-12 21:14:14

import os
import sys
from utils import *
import argparse
import numpy as np
from model import *
from keras.utils import np_utils
from metric import get_ner_fmeasure
from extractor import extractor_embeddings, extractor_weights
from extractor import save_weights, save_bias

np.random.seed(1234)

logger = get_logger('Train segmenation word embedding')


def main():
    parser = argparse.ArgumentParser("Train segmenation embedding ...")

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument('--context_size', type=int, default=2, help='Context size')
    parser.add_argument('--hidden_units', type=int, default=150, help='Num of units in hidden layer')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate for layers')
    parser.add_argument('--train', type=str, help='train data path')
    parser.add_argument('--test', type=str, help='test data path')
    parser.add_argument('--uni_vocab', type=str, help='unigram vocabulary path')
    parser.add_argument('--bi_vocab', type=str, help='bigram vocabulary path')
    parser.add_argument('--uni_embed', type=str, help='unigram embedding path')
    parser.add_argument('--bi_embed', type=str, help='bigram embedding path')
    parser.add_argument('--embed_dim', type=int, default=50, help='embedding dimension')
    parser.add_argument('--modelpath', type=str, help='checkpoint/path to save model')
    parser.add_argument('--train_flag', action='store_true', help='Train or eval')
    parser.add_argument('--out_embed1', type=str, help='path to save fine-tuned unigram embedding')
    parser.add_argument('--out_embed2', type=str, help='path to save fine-tuned bigram embedding')
    parser.add_argument('--weight_dir', type=str, help='directory to save hidden layer weights')
    parser.add_argument('--load_biEmb', action='store_true', help='load bigram pretrained embedding or not')

    args = parser.parse_args()
    bsize = args.batch_size
    trainpath = args.train
    testpath = args.test

    uni_vocabpath = args.uni_vocab
    bi_vocabpath = args.bi_vocab
    uni_embpath = args.uni_embed
    bi_embpath = args.bi_embed

    # load data
    X1_train, X2_train, X1_test, X2_test,  y_train, y_test, uni_vocab, bi_vocab, label_vocab, uni_embed_table, bi_embed_table, train_sentlenL, test_sentlenL = \
         creat_data(trainpath, testpath, args.context_size, uni_vocabpath, bi_vocabpath, args.embed_dim, uni_embpath, bi_embpath, load_biEmb=args.load_biEmb)

    idx2uni = {v: k for k,v in uni_vocab.iteritems()}
    idx2bi = {v: k for k,v in bi_vocab.iteritems()}

    # print test bichars: TODO
    # X2L = X2_test.tolist()
    # for i in xrange(len(X2L)):
    #     print "instance %s" % str(i+1)
    #     bichars = [idx2bi[j] for j in X2L[i]]
    #     _bichars = " ".join(bichars)
    #     print X2L[i]
    #     print _bichars

    num_classes = len(label_vocab)
    uni_vocabsize = len(uni_vocab)
    bi_vocabsize = len(bi_vocab)
    window_size = 2 * args.context_size + 1

    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    # for debugging
    print X1_train[0]
    print X2_train[0]
    print Y_train[0]
    print X1_train.shape, Y_train.shape

    if not args.load_biEmb:
        bi_embed_table = None
    model = build_model(args, uni_vocabsize, bi_vocabsize, window_size, num_classes, uni_embed_table, bi_embed_table, verbose=True)
    modelpath = args.modelpath
    if args.train_flag:
        logger.info("Train the model")
        checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1, save_best_only=True)
        model.fit([X1_train, X2_train], [Y_train], nb_epoch=args.num_epochs, batch_size=args.batch_size, validation_data=([X1_test, X2_test], [Y_test]), callbacks=[checkpointer], shuffle=True)

    model.load_weights(modelpath)
    pred_Y_test = model.predict([X1_test, X2_test], batch_size=32)
    test_argmax = np.argmax(pred_Y_test, axis=1)
    assert test_argmax.shape[0] == Y_test.shape[0]

    def get_sentlist(sentlenList, L):
        idx = 0
        sentL = []
        for slen in sentlenList:
            # print idx
            cur_sent = L[idx:slen+idx]
            idx += slen
            sentL.append(cur_sent)
        # print idx, len(L)
        assert idx == len(L)
        return sentL

    def get_labels(index_sentL, label_vocab):
        idx_labels = {v: k for k, v in label_vocab.iteritems()}
        cur_index = 0

        labelList = []
        for l in index_sentL:
            labels = []
            for idx in l:
                labels.append(idx_labels[idx])
            labelList.append(labels)
        # print labelList
        return labelList
    # test evaluation
    pred_sent_indexL = get_sentlist(test_sentlenL, test_argmax.tolist())
    gold_sent_indexL = get_sentlist(test_sentlenL, y_test)

    # get sent list of labels
    pred_sentL = get_labels(pred_sent_indexL, label_vocab)
    gold_sentL = get_labels(gold_sent_indexL, label_vocab)
    # print test_sentlenL
    # print len(gold_sentL[0]), len(gold_sentL[-1]), len(gold_sentL)
    # for i in xrange(len(pred_testList)):
    #     for j in xrange(len(pred_testList[i])):
    #         print gold_testList[i][j], pred_testList[i][j]

    prec, recall, fmeasure = get_ner_fmeasure(gold_sentL, pred_sentL)
    logger.info("Precision = %s, recall = %s, F-measure = %s " % (prec, recall, fmeasure))

    uni_embeddings, bi_embeddings = extractor_embeddings(model, uni_vocabpath, bi_vocabpath, args.out_embed1, args.out_embed2)
    Wh1, bh1 = extractor_weights(model, 'h')
    Wh2, bh2 = extractor_weights(model, 'new_h')
    weight_path = os.path.join(args.weight_dir, "hidden_wb.txt")
    # wh1_path = os.path.join(args.weight_dir, 'hidden1_W.txt')
    # bh1_path = os.path.join(args.weight_dir, 'hidden1_b.txt')
    # wh2_path = os.path.join(args.weight_dir, 'hidden2_W.txt')
    # bh2_path = os.path.join(args.weight_dir, 'hidden2_b.txt')

    save_weights([Wh1, bh1, Wh2, bh2], weight_path)
    # save_bias(bh1, bh1_path)
    # save_weights(Wh2, wh2_path)
    # save_bias(bh2, bh2_path)

    # print the outputs of hidden states
    # layernames = ['x1', 'x2', 'merged', 'h', 'new_h', 'out']
    # for lname in layernames:
    #     print "layer name %s " % lname
    #     layerout = get_layer_outputs(model, lname, [X1_test[0:2], X2_test[0:2]])
    #     print layerout


if __name__ == '__main__':
    main()
