# -*- coding: utf-8 -*-
# @Author: feidong
# @Date:   2017-01-02 15:04:57
# @Last Modified by:   feidong
# @Last Modified time: 2017-01-12 20:44:05

import codecs
import logging
import numpy as np

logger = logging.getLogger(__name__)

class W2VEmbReader:
    def __init__(self, emb_path, vocab, emb_dim=None):
        logger.info('Loading embeddings from: ' + emb_path)
        has_header=False
        with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
            tokens = emb_file.next().split()
            if len(tokens) == 2:
                try:
                    int(tokens[0])
                    int(tokens[1])
                    has_header = True
                except ValueError:
                    pass
        if has_header:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                tokens = emb_file.next().split()
                assert len(tokens) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, emb_dim)'
                self.vocab_size = int(tokens[0])
                self.emb_dim = int(tokens[1])
                assert self.emb_dim == emb_dim, 'The embeddings dimension does not match with the requested dimension'
                self.embeddings = {}
                counter = 0
                for line in emb_file:
                    tokens = line.split()
                    assert len(tokens) == self.emb_dim + 1, 'The number of dimensions does not match the header info'
                    word = tokens[0]
                    vec = tokens[1:]
                    if word in vocab:
                        self.embeddings[word] = vec
                    counter += 1
                assert counter == self.vocab_size, 'Vocab size does not match the header info'
        else:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                self.vocab_size = 0
                self.emb_dim = -1
                self.embeddings = {}
                for line in emb_file:
                    tokens = line.split()
                    if self.emb_dim == -1:
                        self.emb_dim = len(tokens) - 1
                        assert self.emb_dim == emb_dim, 'The embeddings dimension does not match with the requested dimension'
                    else:
                        assert len(tokens) == self.emb_dim + 1, 'The number of dimensions does not match the header info'
                    word = tokens[0]
                    vec = tokens[1:]
                    if word in vocab:
                        self.embeddings[word] = vec
                    self.vocab_size += 1
        
        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))
    
    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None
    
    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        normalize = True
        for word, index in vocab.iteritems():
            try:
                emb_matrix[index] = self.embeddings[word]
                # print index, word, self.embeddings[word]
                counter += 1
            except KeyError:
                # if word == '-padding-':
                #     emb_matrix[index] = 0.0
                # else:
                scale = np.sqrt(3.0 / self.emb_dim)
                emb_matrix[index] = np.random.uniform(-scale, scale, [1, self.emb_dim])
                pass
        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))

        if normalize:
            # emb_matrix.astype('float32')
            normSum = np.sqrt(np.square(emb_matrix).sum(axis=1, keepdims=True))
            for i in xrange(0, emb_matrix.shape[0]):
                emb_matrix[i] = emb_matrix[i] / normSum[i]
        # print np.square(emb_matrix[3, :]).sum()
        # print np.sum(emb_matrix[2, :])
        # print emb_matrix
        emb_matrix[0] = 0.0
        return emb_matrix
    
    def get_emb_dim(self):
        return self.emb_dim