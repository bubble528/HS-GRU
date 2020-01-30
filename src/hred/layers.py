import tensorflow as tf
import numpy as np

import initializer


def embedding_layer(x, name='embedding-layer', vocab_dim=90004, embedding_dim=256, reuse=None):
    """
    Used before the query encoder, to go from the vocabulary to an embedding
    """

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(name="weights", shape=(vocab_dim, embedding_dim),
                            initializer=tf.random_normal_initializer(stddev=0.01))
        embedding = tf.nn.embedding_lookup(W, x)

    return embedding


def gru_layer_with_reset(h_prev, x_packed, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for the query encoder layer. The encoder is reset after an EoQ symbol
    has been reached.

    :param h_prev: previous state of the GRU layer
    :param x_packed: x_packed should be a 2-tuple: (embedding, reset vector = x-mask)
    :return: updated hidden layer and reset hidden layer
    """

    # Unpack mandatory packed force_reset_vector, x = embedding
    x, reset_vector = x_packed
    batch_size = tf.shape(x)[0]
    # print(batch_size)
    # print(x)
    # print(reset_vector)

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim, reuse=reuse)

        # print(h)
        # Force reset hidden state: is set to zero if reset vector consists of zeros
        # h_reset = reset_vector * h
        # print(type(reset_vector))
        # reset_vector = tf.tile(tf.reshape(reset_vector, [batch_size, 1]), multiples=[1, y_dim])
        # h_reset = tf.multiply(reset_vector, h)
        h_reset = h
        # h_reset = tf.transpose(tf.matmul(tf.transpose(h), reset_vector))#
        # print(tf.shape(h))
        # print(tf.shape(h_reset))

    return tf.stack([h, h_reset])


def gru_layer_with_retain(h_prev, x_packed, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for the session encoder layer. The current state of the session encoder
    should be retained if no EoQ symbol has been reached yet.
    :param h_prev: previous state of the GRU layer
    :param x_packed: x_packed should be a 2-tuple (embedding, retain vector = x-mask)
    """

    # Unpack mandatory packed retain_vector
    x, retain_vector = x_packed
    batch_size = tf.shape(x)[0]

    with tf.variable_scope(name):
        h = _gru_layer(h_prev, x, 'gru', x_dim, y_dim, reuse=reuse)

        # Force reset hidden state: is h_prev is retain vector consists of ones,
        # is h if retain vector consists of zeros
        retain_vector = tf.tile(tf.reshape(retain_vector, [batch_size, 1]), multiples=[1, y_dim])
        h_retain = retain_vector * h_prev + tf.subtract(np.float32(1.0), retain_vector) * h

    return tf.stack([h, h_retain])


def gru_layer_with_state_reset(h_prev, x_packed, name='gru', x_dim=256, h_dim=512, y_dim=1024, reuse=None):
    """
    Used for the decoder layer
    :param h_prev: previous decoder state
    :param x_packed: should be a 3-tuple (embedder, mask, session_encoder)
    """

    # h_prev = tf.Print(h_prev, [h_prev[:, 1, :]], message="hidden_query: ", summarize=20)

    # Unpack mandatory packed retain_vector and the state
    # x = embedder, ratain_vector = mask, state = session_encoder
    x, retain_vector, state = x_packed
    batch_size = tf.shape(x)[0]

    with tf.variable_scope(name):

        with tf.variable_scope('state_start', reuse=reuse):
            W = tf.get_variable(name='weight', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            # b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.0))
            b = tf.get_variable(name='bias', shape=(y_dim,),
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

            retain_vector = tf.tile(tf.reshape(retain_vector, [batch_size, 1]), multiples=[1, y_dim])
            h_prev_state = retain_vector * h_prev + tf.subtract(np.float32(1.0), retain_vector) * tf.tanh(tf.matmul(state, W) + b)

        h = _gru_layer(h_prev_state, x, 'gru', x_dim, y_dim, reuse=reuse)

    return h


def output_layer(x, h, name='output', x_dim=256, y_dim=512, h_dim=512, reuse=None):
    """
    Used after the decoder
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name, reuse=reuse):
        Wh = tf.get_variable(name='weight_hidden', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias_input', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))
        # b = tf.get_variable(name='bias_input', shape=(y_dim,),
        #                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

        y = tf.matmul(h, Wh) \
            + tf.matmul(x, Wi) \
            + b

    return y


def output_layer_with_state_bias(x, h, state, name='output', x_dim=256, y_dim=512, h_dim=512, s_dim=512, reuse=None):
    """
    Used after the decoder
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name, reuse=reuse):
        Wh = tf.get_variable(name='weight_hidden', shape=(h_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Ws = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        # b = tf.get_variable(name='bias_input', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias_input', shape=(y_dim,),
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

        y = tf.matmul(h, Wh) \
            + tf.matmul(state, Ws) \
            + tf.matmul(x, Wi) \
            + b

    return y


def logits_layer(x, l2_loss, name='logits', x_dim=512, y_dim=90004, reuse=None):
    """
    Used to compute the logits after the output layer.
    The logits could be fed to a softmax layer

    :param x: output (obtained in layers.output_layer)
    :return: logits
    """

    with tf.variable_scope(name, reuse=reuse):

        W = tf.get_variable(name='weight', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.random_normal_initializer(stddev=0.01))
        # b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)

        y = tf.matmul(x, W) + b

    return y, l2_loss


def _gru_layer(h_prev, x, name='gru', x_dim=256, y_dim=512, reuse=None):
    """
    Used for both encoder layers
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate', reuse=reuse):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            # b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
            b_r = tf.get_variable(name='bias', shape=(y_dim,),
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            r = tf.sigmoid(tf.matmul(x, Wi_r) + tf.matmul(h_prev, Wh_r) + b_r)

        # Update gate
        with tf.variable_scope('update_gate', reuse=reuse):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            # b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
            b_z = tf.get_variable(name='bias', shape=(y_dim,),
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            z = tf.sigmoid(tf.matmul(x, Wi_z) + tf.matmul(h_prev, Wh_z) + b_z)

        # Candidate update
        with tf.variable_scope('candidate_update', reuse=reuse):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            # b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(1.0))
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,),
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde) + tf.matmul(r * h_prev, Wh_h_tilde) + b_h_tilde)

        # Final update
        h = tf.subtract(np.float32(1.0), z) * h_prev + z * h_tilde
        # print(tf.shape(r))

    return h


def _rnn_layer(h_prev, x, name='rnn', x_dim=256, y_dim=512, reuse=None):
    """
    Used for both encoder layers,
    this was used for debug purposes
    """

    with tf.variable_scope(name, reuse=reuse):

        Wi = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
        Wh = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
        # b = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
        b = tf.get_variable(name='bias', shape=(y_dim,),
                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

        h = tf.tanh(tf.matmul(x, Wi) + tf.matmul(h_prev, Wh) + b)

    return h


def _gru_layer_with_state_bias(h_prev, x, state, name='gru', x_dim=256, y_dim=1024, s_dim=512, reuse=None):
    """
    Used for decoder. In this GRU the state of the session encoder layer is used when
    computing the decoder updates.
    This is used for "full" state bias in the decoder which we did not use in the end.
    """

    with tf.variable_scope(name):

        # Reset gate
        with tf.variable_scope('reset_gate', reuse=reuse):
            Wi_r = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_r = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_r = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b_r = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
            r = tf.sigmoid(tf.matmul(x, Wi_r) + tf.matmul(h_prev, Wh_r) + tf.matmul(state, Ws_r) + b_r)

        # Update gate
        with tf.variable_scope('update_gate', reuse=reuse):
            Wi_z = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_z = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_z = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            b_z = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
            z = tf.sigmoid(tf.matmul(x, Wi_z) + tf.matmul(h_prev, Wh_z) + tf.matmul(state, Ws_z) + b_z)

        # Candidate update
        with tf.variable_scope('candidate_update', reuse=reuse):
            Wi_h_tilde = tf.get_variable(name='weight_input', shape=(x_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            Wh_h_tilde = tf.get_variable(name='weight_hidden', shape=(y_dim, y_dim), initializer=initializer.orthogonal_initializer(0.01))
            Ws_h_tilde = tf.get_variable(name='weight_state', shape=(s_dim, y_dim), initializer=tf.random_normal_initializer(stddev=0.01))
            # b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,), initializer=tf.constant_initializer(0.1))
            b_h_tilde = tf.get_variable(name='bias', shape=(y_dim,),
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            h_tilde = tf.tanh(tf.matmul(x, Wi_h_tilde) + \
                      tf.matmul(r * h_prev, Wh_h_tilde) + \
                      tf.matmul(state, Ws_h_tilde) + \
                      b_h_tilde)

        # Final update
        h = tf.sub(np.float32(1.0), z) * h_prev + z * h_tilde

    return h


def attention_session(query_encoder_expanded, flatten_decoder, enc_dim=256, dec_dim=256, reuse=None):
    """

    :param query_encoder_expanded: 4D tensor
                                   1) number of steps to attend over OR max steps
                                   2) batch size
                                   3) number of steps
                                   4) query dim
    :param flatten_decoder:
    :param x_dim: dimensionality of query encoder
    :param reuse:
    :return:
    """

    num_of_steps = tf.shape(query_encoder_expanded)[0]
    batch_size = tf.shape(query_encoder_expanded)[1]
    # print(num_of_steps)
    # print(batch_size)

    with tf.variable_scope('attention', reuse=reuse):
        # flatten for eventual multiplication (batch_size + num_of_steps + num_of_steps) x (query_dim)
        flatten_query_encoder_expanded = tf.reshape(query_encoder_expanded, (-1, enc_dim))
        # print("flatten_query_encoder_expanded=")
        # print(flatten_query_encoder_expanded)

        # decoder_dim x query_dim
        W = tf.get_variable(name='weight', shape=(dec_dim, enc_dim),
                            initializer=tf.random_normal_initializer(stddev=0.01))

        # (batch_size + num_of_steps) x (batch_size + num_of_steps + num_of_steps)
        flatten_score = tf.matmul(flatten_decoder, tf.matmul(W, tf.transpose(flatten_query_encoder_expanded)))
        # print("flatten_score=")
        # print(flatten_score)

        # batch_size x num_of_steps x batch_size x num_of_steps x num_of_steps
        score = tf.reshape(flatten_score, (num_of_steps, batch_size, num_of_steps, batch_size, num_of_steps))
        # print("score=")
        # print(score)
        # score = tf.Print(score, [tf.shape(score)])

        # 0:batch_size x 1:num_of_steps x 2:num_of_steps_at
        score = tf.transpose(
            # 0:batch_size x 1:num_of_steps_at x 2:num_of_steps
            tf.matrix_diag_part(
                # 0:batch_size x 1:num_of_steps_at x 2:num_of_steps x 3:num_of_steps
                tf.transpose(
                    # 0:num_of_steps x 1:num_of_steps x 2:num_of_steps_at x 3:batch_size
                    tf.matrix_diag_part(
                        # 0:num_of_steps x 1:num_of_steps x 2:num_of_steps_at x 3:batch_size x 4:batch_size
                        tf.transpose(score, [1, 3, 4, 0, 2])
                    ), [3, 2, 0, 1]
                )
            ), [0, 2, 1]
        )

        # batch_size x num_of_steps x batch_size x num_of_steps x num_of_steps
        a = tf.nn.softmax(score)
        a_broadcasted = tf.tile(tf.expand_dims(a, 3), (1, 1, 1, enc_dim))
        # a_broadcasted = tf.Print(a_broadcasted, [tf.shape(a_broadcasted)])
        context = tf.reduce_sum(a_broadcasted * query_encoder_expanded, 2)
        # context = tf.Print(context, [tf.shape(context)])

        flatten_context = tf.reshape(context, (-1, enc_dim))
        # print(flatten_context)
        # print(flatten_decoder)

    flatten_decoder_with_attention = tf.concat([flatten_context, flatten_decoder], 1)
    # flatten_decoder_with_attention = tf.Print(flatten_decoder_with_attention, [tf.shape(flatten_decoder_with_attention)])
    # print(flatten_decoder_with_attention)

    return flatten_decoder_with_attention


def attention_step(query_encoder_expanded, flatten_decoder, enc_dim=256, dec_dim=256, reuse=None):
    """

    :param query_encoder_expanded: 4D tensor
                                   1) number of steps to attend over OR max steps
                                   2) batch size
                                   3) number of steps
                                   4) query dim
    :param flatten_decoder:
    :param x_dim: dimensionality of query encoder
    :param reuse:
    :return:
    """

    num_of_steps = tf.shape(query_encoder_expanded)[1]
    batch_size = tf.shape(query_encoder_expanded)[0]


    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        # flatten for eventual multiplication (batch_size + num_of_steps) x (query_dim)
        flatten_query_encoder_expanded = tf.reshape(query_encoder_expanded, (-1, enc_dim))

        # decoder_dim x query_dim
        W = tf.get_variable(name='weight', shape=(dec_dim, enc_dim),
                            initializer=tf.random_normal_initializer(stddev=0.01))

        # (batch_size) x (batch_size + num_of_steps)
        flatten_score = tf.matmul(flatten_decoder, tf.matmul(W, tf.transpose(flatten_query_encoder_expanded)))

        # (batch_size) x batch_size x num_of_steps
        score = tf.reshape(flatten_score, (batch_size, batch_size, num_of_steps))

        # (batch_size) x (num_of_steps)
        score = tf.transpose(
            tf.matrix_diag_part(
                tf.transpose(score, [2, 0, 1])
            ), [1, 0]
        )

        a = tf.nn.softmax(score)
        a_broadcasted = tf.tile(tf.expand_dims(a, 2), (1, 1, enc_dim))

        context = tf.reduce_sum(a_broadcasted * query_encoder_expanded, 1)
        # context = tf.Print(context, [tf.shape(context)])

        flatten_context = tf.reshape(context, (-1, enc_dim))

    flatten_decoder_with_attention = tf.concat([flatten_context, flatten_decoder], 1)

    return flatten_decoder_with_attention


def get_self_attention(H, hidden_size, max_length, batch_size, d_a_size=350, r_size=5, reuse=None):
    # H shape : shape: m x b x m x (2*hidden_size)
    initializer = tf.contrib.layers.xavier_initializer()
    H_reshape = tf.reshape(H, [-1, 2 * hidden_size])

    with tf.variable_scope('self_attention', reuse=reuse):

        with tf.name_scope("self-attention"):
            W_s1 = tf.get_variable("W_s1", shape=[2 * hidden_size, d_a_size], initializer=initializer)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, W_s1))
            W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
            _H_s2 = tf.matmul(_H_s1, W_s2)  #  shape: m x b x m x r_size
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [max_length, batch_size, max_length, r_size]), (0, 1, 3, 2))
            A = tf.nn.softmax(_H_s2_reshape, name="count_attention")
            #  A shape: m x b x r_size x m

        with tf.name_scope("sentence-embedding"):
            M = tf.matmul(A, H)
            #  M shape: m x b x r_size x (2*hidden_size)

        with tf.name_scope("fully-connected"):
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size * r_size, hidden_size], initializer=initializer)
            b_fc = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b_fc")
            fc = tf.tanh(tf.nn.xw_plus_b(M_flat, W_fc, b_fc))
            fc_reshape = tf.reshape(fc, (max_length, batch_size, hidden_size))
            #  fc shape: m x b x hidden_size

        with tf.name_scope("penalization"):
            #  AA_T shape: m x b x r_size x r_size
            AA_T = tf.matmul(A, tf.transpose(A, [0, 1, 3, 2]))
            I = tf.reshape(tf.tile(tf.eye(r_size), [max_length * batch_size, 1]), [max_length, batch_size, r_size, r_size])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))

    return fc_reshape, P

def Hidden_query_transform(H, max_length, batch_size, old_dim, new_dim, reuse=None):
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('hidden_Transform', reuse=reuse):
        H_reshape = tf.reshape(H, shape=[-1, old_dim])
        W1 = tf.get_variable("W_t", shape=[old_dim, new_dim], initializer=initializer)
        b1 = tf.Variable(tf.constant(0.1, shape=[new_dim]), name="b_t")
        H_new = tf.nn.xw_plus_b(H_reshape, W1, b1)
        H_new_reshape = tf.reshape(H_new, (max_length, batch_size, new_dim))

    return H_new_reshape

def Output_query_transform(H, max_length, batch_size, old_dim, new_dim, reuse=None):
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('Output_Transform', reuse=reuse):
        H_reshape = tf.reshape(H, shape=[-1, old_dim])
        W1 = tf.get_variable("W_t", shape=[old_dim, new_dim], initializer=initializer)
        b1 = tf.Variable(tf.constant(0.1, shape=[new_dim]), name="b_t")
        H_new = tf.tanh(tf.nn.xw_plus_b(H_reshape, W1, b1))
        H_new_reshape = tf.reshape(H_new, (max_length, batch_size, new_dim))

    return H_new_reshape

def gnn_layer(items_embedder, A, in_size, out_size, step, reuse=None):
    max_length = tf.shape(items_embedder)[0]
    batch_size = tf.shape(items_embedder)[1]
    n_node = tf.shape(items_embedder)[2]
    # initializer = tf.contrib.layers.xavier_initializer()

    fin_state = items_embedder
    cell = tf.nn.rnn_cell.GRUCell(out_size)

    with tf.variable_scope('gru_gnn', reuse=reuse):
        W = tf.get_variable('W_gnn', shape=[out_size, out_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable('b_gnn', [out_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        W_f = tf.get_variable('Wf_gnn', shape=[in_size, out_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        fin_state = tf.matmul(tf.reshape(fin_state, [-1, in_size]), W_f)
        for i in range(step):
            fin_state = tf.reshape(fin_state, [max_length, batch_size, -1,  out_size])
            fin_state_a = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, out_size]), W) + b,
                                   [max_length, batch_size, n_node, out_size])
            a_h = tf.matmul(A, fin_state_a)
            state_output, fin_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(a_h, [-1, out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, out_size]))

        state = tf.reshape(fin_state, [max_length, batch_size, n_node, out_size])
        # 全连接
        # W_fc = tf.get_variable("W_fc", shape=[n_node * out_size, out_size], initializer=initializer)
        # b_fc = tf.Variable(tf.constant(0.1, shape=[out_size]), name="b_fc")
        # state_fc = tf.nn.tanh(tf.nn.xw_plus_b(state, W_fc, b_fc), name="fc")
        state_all = tf.reduce_sum(state, 2)/tf.cast(n_node, tf.float32)

    return state_all


def gnn_attention(session_encoder, attention_mask, aln_encoder, session_dim = 512, gnn_dim = 256, reuse=None):
    max_length = tf.shape(session_encoder)[0]
    batch_size = tf.shape(session_encoder)[1]
    session_encoder_expended = tf.tile(tf.expand_dims(session_encoder, 2), (1, 1, max_length, 1))
    session_encoder_expended = session_encoder_expended * tf.tile(tf.expand_dims(attention_mask, 3),
                                                                  (1, 1, 1, session_dim))
    with tf.variable_scope('session_attention', reuse=reuse):
        W = tf.get_variable(name='weight', shape=(gnn_dim, session_dim),
                            initializer=tf.random_normal_initializer(stddev=0.01))
        aln_encoder_reshape = tf.reshape(tf.matmul(tf.reshape(aln_encoder, (-1, gnn_dim)), W), (max_length, batch_size, -1))
        aln_encoder_reshape = tf.expand_dims(aln_encoder_reshape, 3)
        score = tf.nn.tanh(tf.matmul(session_encoder_expended, aln_encoder_reshape))
        score = tf.reshape(score, (max_length, batch_size, -1))

        score_a = tf.expand_dims(tf.nn.softmax(score), 3)

        session_context = tf.reduce_sum((score_a * session_encoder_expended), 2)

    return session_context
