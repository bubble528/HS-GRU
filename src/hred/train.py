""" File to build and train the entire computation graph in tensorflow
"""

import numpy as np
import tensorflow as tf
import subprocess

tf.logging.set_verbosity(tf.logging.DEBUG) # test

from hred1 import HRED
from optimizer import Optimizer
import _pickle as cPickle
import math
import data_iterator as sordoni_data_iterator  #
from utils1 import *
import networkx as nx

from get_batch import get_batch
import random

from datetime import datetime
import logging

logging.basicConfig(filename='output.log', level=logging.DEBUG)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto(
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

VALIDATION_FILE = '../../data/val_session.out'
TEST_FILE = '../../data/test_session.out'
LOGS_DIR = '../../logs'
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
RESTORE = False

N_BUCKETS = 20
MAX_ITTER = 10000000


CHECKPOINT_FILE = './../../checkpoints/model-huge-attention-fixed.ckpt'#model-huge-attention-fixed.ckpt
# OUR_VOCAB_FILE = '../../data/aol_vocab_50000.pkl'
# OUR_TRAIN_FILE = '../../data/aol_sess_50000.out'
# OUR_SAMPLE_FILE = '../../data/sample_aol_sess_50000.out'
SORDONI_VOCAB_FILE = '../../guonei_data/train.dict.pkl'
SORDONI_TRAIN_FILE = '../../guonei_data/train.ses.pkl'
SORDONI_VALID_FILE = '../../guonei_data/valid.ses.pkl'
ALN_PATH = '../../guonei_data/ALN_result_sort_0.1.csv'
VOCAB_SIZE = 3538
EMBEDDING_DIM = 512
QUERY_DIM = 512
SESSION_DIM = 512
BATCH_SIZE = 32
MAX_LENGTH = 100
D_A_SIZE = 350
R_SIZE = 5
P_COEF = 1.0
STEP = 1

# CHECKPOINT_FILE = '../../checkpoints/model-small.ckpt'
# OUR_VOCAB_FILE = '../../data/aol_vocab_2500.pkl'
# OUR_TRAIN_FILE = '../../data/small_train.out'
# OUR_SAMPLE_FILE = '../../data/sample_small_train.out'
# SORDONI_VOCAB_FILE = '../../data/sordoni/dev_large/train.dict.pkl'
# SORDONI_TRAIN_FILE = '../../data/sordoni/dev_large/train.ses.pkl'
# SORDONI_VALID_FILE = '../../data/sordoni/dev_large/valid.ses.pkl'
# VOCAB_SIZE = 2504
# EMBEDDING_DIM = 10
# QUERY_DIM = 15
# SESSION_DIM = 20
# BATCH_SIZE = 80
# MAX_LENGTH = 50
SEED = 1234


class Trainer(object):
    def __init__(self):

        vocab = cPickle.load(open(SORDONI_VOCAB_FILE, 'rb'))
        self.vocab_lookup_dict = {k: v for v, k in vocab}  #

        self.train_data, self.valid_data = sordoni_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': N_BUCKETS,
            'bs': BATCH_SIZE,
            'train_session': SORDONI_TRAIN_FILE,
            'seqlen': MAX_LENGTH,
            'valid_session': SORDONI_VALID_FILE
        })

        self.train_data.start()
        self.valid_data.start()

        vocab_size = len(self.vocab_lookup_dict)

        # self.graph = build_aln_graph(ALN_PATH)

        # self.train_data = cPickle.load(open(SORDONI_TRAIN_FILE, 'rb'))
        # self.valid_data = cPickle.load(open(SORDONI_VALID_FILE, 'rb'))

        # vocab_size = VOCAB_SIZE
        # self.vocab_lookup_dict = read_data.read_vocab_lookup(OUR_VOCAB_FILE)

        self.hred = HRED(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, query_dim=QUERY_DIM,
                         session_dim=SESSION_DIM, decoder_dim=QUERY_DIM, output_dim=EMBEDDING_DIM,
                         eoq_symbol=EOQ_SYMBOL, eos_symbol=EOS_SYMBOL, unk_symbol=UNK_SYMBOL,
                         d_a_size=D_A_SIZE, r_size=R_SIZE, p_coef=P_COEF, step=STEP) #, batch_size=BATCH_SIZE

        batch_size = None  # BATCH_SIZE
        max_length = None  # MAX_LENGTH

        self.X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.attention_mask = tf.placeholder(tf.float32, shape=(max_length, batch_size, max_length))

        self.X_sample = tf.placeholder(tf.int64, shape=(batch_size,))
        self.H_query = tf.placeholder(tf.float32, shape=(None, batch_size, self.hred.query_dim))
        self.H_session = tf.placeholder(tf.float32, shape=(batch_size, self.hred.session_dim))
        self.H_decoder = tf.placeholder(tf.float32, shape=(batch_size, self.hred.decoder_dim))

        self.logits = self.hred.step_through_session(self.X, self.attention_mask)
        self.loss = self.hred.loss(self.X, self.logits, self.Y)
        self.softmax = self.hred.softmax(self.logits)
        self.accuracy = self.hred.non_padding_accuracy(self.logits, self.Y)
        self.non_symbol_accuracy = self.hred.non_symbol_accuracy(self.logits, self.Y)

        self.session_inference = self.hred.step_through_session(
             self.X, self.attention_mask, return_softmax=True, return_last_with_hidden_states=True,
             reuse=True)
        # self.step_inference = self.hred.single_step(
        #      self.X_sample, self.H_query, self.H_session, self.H_decoder, reuse=True
        # )

        self.optimizer = Optimizer(self.loss)
        self.summary = tf.summary.merge_all()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, max_epochs=1000, max_length=50, batch_size=80):

        stop_flag = []
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        # module_file = tf.train.latest_checkpoint(CHECKPOINT_FILE)

        with tf.Session(config=config) as tf_sess:

            if RESTORE:
                # Restore variables from disk.
                # if module_file is not None:
                #     self.saver.restore(tf_sess,  module_file)
                # else:
                #     tf_sess.run(init_op)
                # print("Model restored.")
                self.saver.restore(tf_sess, CHECKPOINT_FILE)
                print("Model restored.")
            else:
                tf_sess.run(init_op)


            summary_writer = tf.summary.FileWriter(LOGS_DIR, tf_sess.graph)

            total_loss = 0.0
            n_pred = 0.0

            valid_total_loss = 0.0
            valid_n_pred = 0.0

            # train_list = list(range(0, len(self.train_data) -batch_size, batch_size))

            # step = int((len(self.train_data)-150)/batch_size)
            for iteration in range(MAX_ITTER+1):  # 20001

                x_batch, y_batch, seq_len = self.get_batch(self.train_data)


                attention_mask = make_attention_mask(x_batch)

                # attention_mask shape : MAX_LENGTH X BATCH_SIZE X MAX_LENGTH
                # A, items = get_gnn_a(x_batch, attention_mask, self.graph)


                if iteration % 10 == 0:

                    x_valid, y_valid, valid_len = self.get_batch(self.valid_data)
                    valid_mask = make_attention_mask(x_valid)

                    logits_out, loss_out, _, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                        [self.logits, self.loss, self.optimizer.optimize_op, self.accuracy, self.non_symbol_accuracy],
                        {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask}
                    )

                    valid_logits_out, valid_loss_out, valid_acc_out, valid_accuracy_non_special_symbols_out = tf_sess.run(
                        [self.logits, self.loss, self.accuracy, self.non_symbol_accuracy],
                        {self.X: x_valid, self.Y: y_valid, self.attention_mask: valid_mask}
                    )


                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp


                    valid_total_loss_tmp = valid_total_loss + valid_loss_out
                    valid_n_pred_tmp = valid_n_pred + valid_len * batch_size
                    valid_cost = valid_total_loss_tmp / valid_n_pred_tmp

                    if iteration % 100 == 0:
                        print("Step %d - Cost: %f   ValidCost: %f   Loss: %f   ValidLoss: %f   Accuracy: %f   ValidAccuracy: %f   Accuracy (no symbols): %f  ValidAccuracy (no symbols): %f  Length: %d  ValidLength: %d" %
                              (iteration, cost, valid_cost, loss_out, valid_loss_out, acc_out, valid_acc_out,
                               accuracy_non_special_symbols_out, valid_accuracy_non_special_symbols_out,
                               seq_len, valid_len))

                    logging.debug(
                         "[{}] Train Step {:d}/{:d} - Cost: {:f}   ValidCost: {:f}   Loss: {:f}   ValidLoss: {:f}   Accuracy: {:f}   ValidAccuracy: {:f}   Accuracy (no symbols): {:f}  ValidAccuracy (no symbols): {:f}  Length: {:d}  ValidLength: {:d}  Batch Size: {:d}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), iteration, MAX_ITTER,
                            cost, valid_cost, loss_out, valid_loss_out, acc_out, valid_acc_out,
                            accuracy_non_special_symbols_out, valid_accuracy_non_special_symbols_out, seq_len,
                            valid_len, batch_size
                         ))

                else:
                    loss_out, _ = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op],
                        {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp

                if math.isnan(loss_out) or math.isnan(cost) or cost > 100:
                    print("Found inconsistent results, restoring model...")
                    self.saver.restore(tf_sess, CHECKPOINT_FILE)
                else:
                    total_loss = total_loss_tmp
                    n_pred = n_pred_tmp

                    if iteration % 100 == 0:
                        if len(stop_flag) == 10:
                            if max(stop_flag) < cost:
                                break
                            else:
                                stop_flag.pop(0)
                                stop_flag.append(cost)
                        else:
                            stop_flag.append(cost)

                        print("Saving..")
                        self.save_model(tf_sess, loss_out, iteration)


                # Sumerize
                if iteration % 100 == 0:
                #if iteration % 100 == 0:
                    summary_str = tf_sess.run(self.summary, {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()
                #
                # if iteration % 500 == 0:
                #      # self.sample(tf_sess)
                #      self.sample_beam(tf_sess)

                iteration += 1


    def sample(self, sess, max_sample_length=30, num_of_samples=3, min_queries = 3):

        for i in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len // 2), 1], axis=1)

            attention_mask = make_attention_mask(input_x)
            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x, self.attention_mask: attention_mask}
            )

            queries_accepted = 0
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

            # Ignore UNK and EOS (for the first min_queries)
            arg_sort_i = 0
            while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                            arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                arg_sort_i += 1
            x = arg_sort[arg_sort_i]

            if x == self.hred.eoq_symbol:
                queries_accepted += 1

            result = [x]
            i = 0

            while x != self.hred.eos_symbol and i < max_sample_length:
                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder}
                )
                print("INFO -- Sample hidden states", tf.shape(hidden_query))
                arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

                # Ignore UNK and EOS (for the first min_queries)
                arg_sort_i = 0
                while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                    arg_sort_i += 1
                x = arg_sort[arg_sort_i]

                if x == self.hred.eoq_symbol:
                    queries_accepted += 1

                result += [x]
                i += 1

            input_x = np.array(input_x).flatten()
            result = np.array(result).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))
            print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))
            print('')

    def sample_beam(self, sess, max_steps=30, num_of_samples=3, beam_size=10, min_queries=2):

        # valid_list = list(range(0, len(self.valid_data) - 150, BATCH_SIZE))
        for step in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            # random_element = random.choice(valid_list)
            # x_batch, y_batch, seq_len, train_list = get_batch(valid_list, self.valid_data, type='train',
            #                                                   element=random_element,
            #                                                   batch_size=BATCH_SIZE,
            #                                                   max_len=MAX_LENGTH,
            #                                                   eoq=EOQ_SYMBOL)

            input_x = np.expand_dims(x_batch[:-(seq_len // 2), 1], axis=1)
            x_batch = np.expand_dims(x_batch[:, 1], axis=1)

            attention_mask= make_attention_mask(input_x)


            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x, self.attention_mask: attention_mask}
            )

            current_beam_size = beam_size
            current_hypotheses = []
            final_hypotheses = []

            # Reverse arg sort (highest prob above)
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]  # 所有词的由生成概率从大到小排序
            arg_sort_i = 0

            # create original current_hypotheses
            while len(current_hypotheses) < current_beam_size:
                # Ignore UNK and EOS (for the first min_queries)
                while arg_sort[arg_sort_i] == self.hred.unk_symbol or arg_sort[arg_sort_i] == self.hred.eos_symbol:
                    arg_sort_i += 1

                x = arg_sort[arg_sort_i]  # 依概率大小取词id
                arg_sort_i += 1

                queries_accepted = 1 if x == self.hred.eoq_symbol else 0
                result = [x]
                prob = softmax_out[0][x]
                current_hypotheses += [
                    (prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted)]

            # Create hypotheses per step
            step = 0
            while current_beam_size > 0 and step <= max_steps:

                step += 1
                next_hypotheses = []

                # expand all hypotheses
                for prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted in current_hypotheses:

                    input_for_mask = np.concatenate((input_x, np.expand_dims(np.array(result), axis=1)), axis=0)
                    attention_mask= make_attention_mask(input_for_mask)


                    softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                        self.session_inference,
                        {self.X: input_for_mask, self.attention_mask: attention_mask}
                    )

                    # Reverse arg sort (highest prob above)
                    arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
                    arg_sort_i = 0

                    expanded_hypotheses = []

                    # create hypothesis
                    while len(expanded_hypotheses) < current_beam_size:

                        # Ignore UNK and EOS (for the first min_queries)
                        while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                arg_sort[
                                    arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                            arg_sort_i += 1

                        new_x = arg_sort[arg_sort_i]
                        arg_sort_i += 1

                        new_queries_accepted = queries_accepted + 1 if x == self.hred.eoq_symbol else queries_accepted
                        new_result = result + [new_x]
                        new_prob = softmax_out[0][new_x] * prob

                        expanded_hypotheses += [(new_prob, new_x, new_result, hidden_query, hidden_session,
                                                 hidden_decoder, new_queries_accepted)]

                    next_hypotheses += expanded_hypotheses

                # sort hypotheses and remove the least likely
                next_hypotheses = sorted(next_hypotheses, key=lambda x: x[0], reverse=True)[:current_beam_size]
                current_hypotheses = []

                for hypothesis in next_hypotheses:
                    _, x, _, _, _, _, queries_accepted = hypothesis

                    if x == self.hred.eos_symbol:
                        final_hypotheses += [hypothesis]
                        current_beam_size -= 1
                    else:
                        current_hypotheses += [hypothesis]

            final_hypotheses += current_hypotheses

            input_x = np.array(input_x).flatten()
            x_batch = np.array(x_batch).flatten()
            print('event link:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in x_batch]),))
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))
            # logging.debug(
            #     "Sample input: {:s}".format(' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))

            for _, _, result, _, _, _, _ in final_hypotheses:
                result = np.array(result).flatten()
                print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))
                # logging.debug(
                #     "Sample output: {:s}".format(' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))

            print('')

    def save_model(self, sess, loss_out, iteration):
        if not math.isnan(loss_out):
            # Save the variables to disk.
            save_path = self.saver.save(sess, save_path=CHECKPOINT_FILE, global_step=iteration)
            print("Model saved in file: %s" % save_path)

    def get_batch(self, train_data):

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

    def get_sentences(self, batch, output_logits, max_length):
        outputs = np.argmax(output_logits, axis=2)
        batch_sentence = [''] * len(batch)
        output_sentence = [''] * len(outputs)
        # total_length = max_length
        for i in range(max_length):
            for j in range(len(batch)):
                batch_sentence[i] += self.vocab_lookup_dict[batch[i][j]] + ' '
                output_sentence[i] += self.vocab_lookup_dict[outputs[i][j]] + ' '
        return np.transpose(batch_sentence), np.transpose(output_sentence)

    def predict_model(self, sess=None, batch_size=80):

        with tf.Session() as tf_sess:
            # tf.Graph().finalize()
            self.saver.restore(tf_sess, './../../checkpoints/model-huge-attention-fixed.ckpt-878100')  # self.config.checkpoint_path
            for iteration in range(20):
                x_batch, y_batch, seq_len = self.get_batch(self.valid_data)
                # print(x_batch)
                # print(y_batch)
                attention_mask = make_attention_mask(x_batch)

                logits_out, loss_out, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                    [self.logits, self.loss, self.accuracy, self.non_symbol_accuracy],
                    {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask}
                )

                output = tf_sess.run(tf.nn.softmax(logits_out))
                cost = loss_out / (seq_len * batch_size)
                # batch_sentences, pred_sentences = self.get_sentences(y_batch, output, seq_len)
                print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f  Length: %d" %
                      (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len))
                # logging.debug(
                #     "[{}] Train Step {:d}/{:d} - Cost: {:f}   Loss: {:f}   Accuracy: {:f}   Accuracy (no symbols): {:f}  Length: {:d}  Batch Size: {:d}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), iteration, MAX_ITTER,
                #         cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len, batch_size
                #     ))

                # print(batch_sentences)
                # print(pred_sentences)
                # logging.debug(batch_sentences)
                # logging.debug(pred_sentences)
                # self.sample_beam(tf_sess)

        return


if __name__ == '__main__':
    with tf.Graph().as_default():
        trainer = Trainer()
        # trainer.train(batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
        trainer.predict_model(batch_size=BATCH_SIZE)
