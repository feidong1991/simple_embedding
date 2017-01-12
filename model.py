# -*- coding: utf-8 -*-
# @Author: feidong
# @Date:   2017-01-02 20:08:21
# @Last Modified by:   feidong
# @Last Modified time: 2017-01-12 20:44:51
# import numpy as np
import os
import time
from utils import get_logger
from keras.regularizers import l2
from keras.callbacks import *
from keras.optimizers import *
# from keras.utils.np_utils import to_categorical, accuracy
from keras.layers.core import *
from keras.layers import Input, Embedding, Dense, merge, Flatten
from keras.models import *
from zeromasking import ZeroMaskedEntries, MaskEatingLambda


np.random.seed(1234)
logger = get_logger("Build model")


def build_model(opts, uni_vocabsize, bi_vocabsize, window_size, num_classes, uni_embed_W, bi_embed_W, verbose=False):
    L = window_size

    input1 = Input(shape=(L, ), name='input1')
    x1 = Embedding(output_dim=opts.embed_dim, input_dim=uni_vocabsize, input_length=L, weights=uni_embed_W, mask_zero=True, name='x1')(input1)

    input2 = Input(shape=(L, ), name='input2')
    x2 = Embedding(output_dim=opts.embed_dim, input_dim=bi_vocabsize, input_length=L, weights=bi_embed_W, mask_zero=True, name='x2')(input2)

    # x1_maskedout = MaskEatingLambda(name='x1_maskedout')(x1)
    # x2_maskedout = MaskEatingLambda(name='x2_maskedout')(x2)
    x1_maskedout = ZeroMaskedEntries(name='x1_maskedout')(x1)
    x2_maskedout = ZeroMaskedEntries(name='x2_maskedout')(x2)
    merged = merge([x1_maskedout, x2_maskedout], mode='concat', name='merged')
    # merged = merge([x1, x2], mode='concat', name='merged')
    drop_out = Dropout(opts.drop_rate, name='dropout')(merged)
    # z = Reshape((2*L*opts.embed_dim, ), name='z')(drop_out)
    z = Flatten(name='z')(drop_out)
    h = Dense(opts.hidden_units, activation='tanh', name='h')(z)
    new_h = Dense(opts.hidden_units, activation='tanh', name='new_h')(h)
    out = Dense(num_classes, activation='softmax', name='out')(new_h)

    model = Model(input=[input1, input2], output=out)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)
    return model


def get_layer_weights(model, layer_name):
    layer = model.get_layer(layer_name)
    Wb = layer.get_weights()
    return Wb


def get_layer_outputs(model, layer_name, data):
        #layer_name = 'my_layer'
        layer_model = Model(input=model.input,
                                         output=model.get_layer(layer_name).output)
        layer_output = layer_model.predict(data)
        return layer_output


def get_embed_vec(model, layer_name, data):
    # input data
    assert layer_name in ['x1', 'x2']
    embedM = get_layer_outputs(model, layer_name, data)
    # print embedM[0, 0, :], embedM[0, 1, :]
    return embedM[:, 0, :]