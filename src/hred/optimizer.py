import tensorflow as tf


class Optimizer():

    def __init__(self, loss, initial_learning_rate=0.0002, num_steps_per_decay=10000,
                          decay_rate=0.5, max_global_norm=1.0):
        """ Create a simple optimizer.

        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.

        Args:
            loss: A 0-D float32 Tensor.
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        """

        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        # learning_rate = tf.train.exponential_decay(
        #     initial_learning_rate, self.global_step, num_steps_per_decay,
        #     decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=1e-6)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.95, momentum=0.9, epsilon=1e-6)

        self._optimize_op = optimizer.apply_gradients(grad_var_pairs, global_step=self.global_step)
        # self._optimize_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)  # , epsilon=1e-6

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op
