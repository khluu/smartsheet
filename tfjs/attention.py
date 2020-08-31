# -*- coding: utf-8 -*-
"""attention

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VHH8L4OaULNQswxYRqbhB5sW1dMDco14
"""

import sys
print(sys.argv[0])
import math
import tensorflow as tf
print(tf.__version__)
import os
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
np.random.seed(123)
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam

from keras.layers import LSTM, GRU, Bidirectional
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Sequential
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import GRU, Attention, Input, Concatenate, Dense, TimeDistributed, Activation, dot

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')
    h5file.close()

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
def convert_tensor(arg):
    return tf.convert_to_tensor(arg, dtype=tf.float32)

from google.colab import drive
drive.mount('/content/drive/')
filename = "/content/drive/My Drive/ColabPro/train_attn2.hdf5"
train_data = load_dict_from_hdf5(filename)

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
unique_labels = {}

def to_lists(data):
    labels_lst = []
    curves_lst = []
    max_curves = 0
    for a in data:
        curves_lst_inside = []
        labels_lst_inside = []
        for b in data[a]:
            label = data[a][b]['label']
            if(label not in unique_labels): unique_labels[label] = 1
            labels_lst_inside.append(label)
            sample = np.concatenate(list(data[a][b]['feat_bez_curves'].values()))
            sample = np.nan_to_num(sample)
            curves_lst_inside.append(sample)
            max_curves = max(max_curves,sample.shape[0])
        curves_lst.append(curves_lst_inside)
        labels_lst.append(labels_lst_inside)
    return curves_lst, labels_lst, max_curves
        
#From here: https://stackoverflow.com/questions/57346556/creating-a-ragged-tensor-from-a-list-of-tensors
def stack_ragged(tensors):
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

def stack_dense(tensors, max_curves):
    pad = lambda x:np.pad(x, pad_width=((0,max_curves - len(x)), (0,0)), constant_values=(-99))
    return np.concatenate([np.expand_dims(pad(x), 0) for x in tensors],axis=0)
def stack_start_end(tensors):
    pad = lambda x:np.pad(x, pad_width=((1,1), (0,0)), constant_values=(-1))
    return np.concatenate([np.expand_dims(pad(x), 0) for x in tensors], axis=0)

def stack_eq(tensors):
    max_count = 0
    for i in range(len(tensors)):
        max_count = max(max_count, tensors[i].shape[0])
    pad = lambda x: np.pad(x, pad_width=((0, max_count - len(x)), (0,0), (0,0)))
    return np.concatenate([np.expand_dims(pad(x),0) for x in tensors],axis=0)

def encode_labels(labels,fit=True):
    if(fit): label_encoder.fit(labels)
    integer_encoded = label_encoder.transform(labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    if(fit): onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)
    return onehot_encoded

def stack_labels_dense(tensors):
    max_count = 0
    for i in tensors:
        max_count = max(max_count, len(i))
    pad = lambda x: np.pad(x, pad_width=((0, max_count - len(x)), (0,0)))
    return np.concatenate([np.expand_dims(pad(x), 0) for x in tensors],axis=0)

#Process train_data
curves_lst, labels_lst, max_curves = to_lists(train_data)
def processData(train_data):
  
  curves = []
  labels = []
  max_curves = 0
  for i in range(len(curves_lst)):
      curves_inside = []
      for j in curves_lst[i]:
          for l in j:
              curves_inside.append(l)
      curves.append(curves_inside)
      max_curves = max(max_curves, len(curves_inside))
  return stack_start_end(stack_dense(curves, max_curves))


def processLabel(train_data):
  labels_lst_common = []
  labels_lst_common.append("<start>")
  labels_lst_common.append("<end>")
  len_memory = []
  for i in labels_lst:
      len_memory.append(len(i))
      for j in i:
          labels_lst_common.append(j)
  labels = encode_labels(np.array(labels_lst_common))
  start = 2
  Y_train = []
  for i in len_memory:
      Y_train.append((labels[start:start+i]))
      start = start+i
  Y_train = np.array(Y_train)
  stack_labels = []
  for i in range(len(labels_lst)):
      labels_inside = []
      labels_inside.append(labels[0])
      for j in range(len(labels_lst[i])):
          for l in range(len(curves_lst[i][j])):
              labels_inside.append(Y_train[i][j])
      labels_inside.append(labels[1])
      stack_labels.append(labels_inside)
  stack_labels = stack_labels_dense(stack_labels)
  return stack_labels

def processData2(train_data):
  
  curves = []
  labels = []
  max_curves = 0
  for i in range(1):
      curves_inside = []
      
      for j in curves_lst[i]:
          curves_inside2 = []
          curves_inside3 = []
          for l in j:
            if(l[8] == 1):
              print('START: ', l)
              curves_inside3.append(l)
            if(l[8] == -1):
              print('END: ', l)
              curves_inside3.append(l)
              curves_inside2.append(curves_inside3)
            if(l[8] == 0):
              print(l) 
              curves_inside3.append(l)
          if curves_inside3:
            curves_inside2.append(curves_inside3)
          print('------------------')
          curves_inside.append(curves_inside2)
      curves.append(curves_inside)
      #max_curves = max(max_curves, len(curves_inside))
  return curves
  #return stack_start_end(stack_dense(curves, max_curves))

#X_train = processData2(train_data)
#print(labels_lst[0])



X_train = processData(train_data)
Y_train = processLabel(train_data)
print(X_train.shape)
print(Y_train.shape)

#max_curves = 159
max_curves = Y_train.shape[1]
max_characters = 40
curve_size = 9
alphabet_size = Y_train.shape[2]
hidden_size = 100

encoder_inputs = Input(batch_shape=(None, max_curves, curve_size), name='encoder_inputs')
decoder_inputs = Input(batch_shape=(None, max_curves, alphabet_size), name='decoder_inputs')


encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, reset_after=False, name='encoder_gru')
encoder_out, encoder_state = encoder_gru(encoder_inputs)
print('Enc_out: ', encoder_out.shape)

decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, reset_after=False, name='decoder_gru')
decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)
print('Dec_out: ', decoder_out.shape)

#attn_layer = Attention(name='attention_layer')
#attn_out = attn_layer([encoder_out, decoder_out])

#Bahdanau Attention
attn_out = dot([encoder_out, decoder_out], axes=[2, 2])
#attn_out = tf.nn.tanh(attn_out)
#V = tf.keras.layers.Dense(541)
#attn_out = V(attn_out)
attn_out = Activation('softmax')(attn_out)

print('Attn: ', attn_out.shape)
context = dot([attn_out, encoder_out], axes=[2,1])
print('Context: ', context.shape)

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, context])
print('Dec_concat_inp: ', decoder_concat_input.shape)
dense = Dense(alphabet_size, activation='softmax', name='softmax_layer')

dense_time = TimeDistributed(dense, name='time_distributed_layer')

decoder_pred = dense_time(decoder_concat_input)
print('Dense: ', decoder_pred.shape)
full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
full_model.compile(optimizer=Adam(lr=1e-2 * 3.33), loss='categorical_crossentropy', metrics=['accuracy'])

keras.utils.plot_model(full_model, "my_first_model.png")

full_model.fit([X_train, Y_train], Y_train, epochs = 10, batch_size = 320)

