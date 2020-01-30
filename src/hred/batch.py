import numpy as np
import tensorflow as tf
import subprocess

tf.logging.set_verbosity(tf.logging.DEBUG) # test

from hred1 import HRED
from optimizer import Optimizer
import _pickle as cPickle
import math
import sordoni.data_iterator as sordoni_data_iterator
from utils1 import make_attention_mask

from get_batch import get_batch
import random

from datetime import datetime
import logging

VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = True

N_BUCKETS = 20
MAX_ITTER = 10000000


CHECKPOINT_FILE = './../../checkpoints/model-huge-attention-fixed.ckpt'#model-huge-attention-fixed.ckpt
SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'
VOCAB_SIZE = 50003
EMBEDDING_DIM = 128
QUERY_DIM = 256
SESSION_DIM = 512
BATCH_SIZE = 3
MAX_LENGTH = 50
SEED = 1234


def get_batch(train_data):
    # The training is done with a trick. We append a special </q> at the beginning of the dialog
    # so that we can predict also the first sent in the dialog starting from the dialog beginning token (</q>).

    data = train_data.next()
    seq_len = data['max_length']
    prepend = np.ones((1, data['x'].shape[1]))
    x_data_full = np.concatenate((prepend, data['x']))
    x_batch = x_data_full[:seq_len]
    y_batch = x_data_full[1:seq_len + 1]
    # print(x_batch.size)
    # print(type(x_batch))
    # if x_batch.size != BATCH_SIZE:
    #     return None

    # x_batch = np.transpose(np.asarray(x_batch))
    # y_batch = np.transpose(np.asarray(y_batch))

    return x_batch, y_batch, seq_len


train_data, valid_data = sordoni_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': N_BUCKETS,
            'bs': BATCH_SIZE,
            'train_session': SORDONI_TRAIN_FILE,
            'seqlen': MAX_LENGTH,
            'valid_session': SORDONI_VALID_FILE
        })

train_data.start()
valid_data.start()

for x in range(0, 10000):
    x_batch, y_batch, seq_len = get_batch(train_data)
    # print(x_batch)
    # print(y_batch)
    print(seq_len)
    # f = open('test.txt', 'a')  # 若是'wb'就表示写二进制文件
    # f.write(str(x_batch))
    # # f.write(y_batch)
    # f.close()

