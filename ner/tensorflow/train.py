# -*- coding: utf-8 -*
import pickle
import sys
import numpy as np
import tensorflow as tf

# self defined models
from Batch import BatchGenerator

with open('../data/Bosondata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    x_valid = pickle.load(inp)
    x_test = pickle.load(inp)
    y_train = pickle.load(inp)
    y_valid = pickle.load(inp)
    y_test = pickle.load(inp)

print('train length: {}'.format(len(x_train)))
print('test length: {}'.format(len(x_test)))
print('word2id length: {}'.format(len(word2id)))

print('Creating the data generator ...')
data_train = BatchGenerator(x_train, y_train, shuffle=True)
data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
data_test = BatchGenerator(x_test, y_test, shuffle=False)
print('Batch data generator created. ')

# Configs
epochs = 4
batch_size = 32
config = {}
config['lr'] = 0.001    # Learning Rate
config['embedding_dim'] = 100
config['sen_len'] = len(x_train[0])     # 60?
config['batch_size'] = batch_size
config['embedding_size'] = len(word2id) + 1     # Total word count
config['tag_size'] = len(tag2id)    # Total tag count
config['pretrained'] = False

embedding_pre = []


if len(sys.argv) == 2 and sys.argv[1] == 'pretrained':
    pass

if len(sys.argv) == 2 and sys.argv[1] == 'test':
    pass

elif len(sys.argv) == 3:
    pass

else:
    print('begin to train...')
    

    print('finish training')