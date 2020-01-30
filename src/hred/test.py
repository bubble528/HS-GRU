import numpy as np
import itertools
from optimizer import Optimizer
import _pickle as cPickle
import math
import data_iterator as sordoni_data_iterator  #
from utils1 import *

def make_attention_mask(X, eoq_symbol=EOQ_SYMBOL):
    def make_mask(last_seen_eoq_pos, query, eoq_symbol):

        if last_seen_eoq_pos == -1:
            return np.zeros((1, len(query)))
        else:
            mask = np.ones([len(query)])
            mask[last_seen_eoq_pos:] = 0.
            mask = np.where(query == eoq_symbol, 0, mask)
            return mask.reshape(1, (len(mask)))

    # eoq_mask = np.where(X == float(EOQ_SYMBOL), float(EOQ_SYMBOL), 0.)
    first_query = True

    for i in range(X.shape[1]):  # loop over batch size --> this gives 80 queries
        query = X[:, i]  # eoq_mask[:, i] #X[:, i]
        # print("query", query)

        first = True
        last_seen_eoq_pos = -1
        # query_masks = []
        for w_pos in range(len(query)):

            if query[w_pos] == float(eoq_symbol):
                last_seen_eoq_pos = w_pos

            if first:
                query_masks = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                first = False
            else:
                new_mask = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                query_masks = np.concatenate((query_masks, new_mask), axis=0)

        global query_masks
        if first_query:
            batch_masks = np.expand_dims(query_masks, axis=2)
            first_query = False
        else:
            batch_masks = np.dstack((batch_masks, query_masks))

    batch_masks = np.transpose(batch_masks, (0, 2, 1))
    # print("shape batch masks", batch_masks.shape)
    # print("batch masks:", batch_masks)
    # print("-------------------------------")
    # print(np.shape())

    return batch_masks  # = attention masks

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


# x_batch = np.array([[5, 3, 4, 1, 5, 3, 4, 1, 4, 6, 6, 1, 2, 0, 0], [4, 5, 1, 3, 4, 6, 7, 1, 4, 6, 6, 5, 1, 2, 0]])
# x_batch = x_batch.T
# batch_mask = make_attention_mask(x_batch)
# print(batch_mask)

SORDONI_VOCAB_FILE = '../../data/financial_data_new/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../data/financial_data_new/train.ses.pkl'
SORDONI_VALID_FILE = '../../data/financial_data_new/valid.ses.pkl'
ALN_PATH = '../../data/ALN_result_sort_0.01.csv'
BATCH_SIZE = 3
MAX_LENGTH = 100
D_A_SIZE = 350
R_SIZE = 5
P_COEF = 1.0
STEP = 1
SEED = 1234
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = True

N_BUCKETS = 20
MAX_ITTER = 10000000

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
aln = get_aln_sort(ALN_PATH)
print("get_ALN")

for iteration in range(3):  # 20001

    x_batch, y_batch, seq_len = get_batch(train_data)
    # for i in range(seq_len):
    #     for j in range(BATCH_SIZE):
    #         print(type(x_batch[i][j]))

    attention_mask = make_attention_mask(x_batch)

    A, items = get_gnn_a(x_batch, attention_mask, aln)
    print(A)
    print(items)
