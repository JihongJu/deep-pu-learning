"""A multilayer perceptron implemented with TensorFlow."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class MultilayerPerceptron(object):
    """MultilayerPerceptron."""

    def __init__(self, n_input, n_classes, n_hiddens=[8, 8],
                 learning_rate=0.01, batch_size=100,
                 training_epochs=30, regularization=1e-3,
                 display_step=5,
                 class_weight=None,
                 imbalanced=None, verbose=None):
        """Init."""
        # Fit Parameters
        self.n_input = n_input
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.regularization = regularization
        self.display_step = display_step
        if class_weight is None:
            self.class_weight = np.ones(n_classes).astype("float")
        else:
            self.class_weight = np.array(class_weight).astype("float")
        if len(self.class_weight) != n_classes:
            raise ValueError("class_weight needs to be of size: {}".format(
                n_classes))
        self.imbalanced = imbalanced
        self.verbose = verbose

        # Define Graph
        tf.reset_default_graph()
        with tf.device('/gpu:0'):
            # tf Graph input
            self.x = tf.placeholder("float", [None, n_input])
            self.y = tf.placeholder("float", [None, n_classes])

            # loss class weight
            self.a = tf.placeholder("float", [n_classes])
            # self.class_weight = tf.ones([n_classes], "float")

            # Layer weights and biases
            self.weights = {
                'h1': tf.get_variable("h1", shape=[n_input, n_hiddens[0]]),
                'h2': tf.get_variable("h2", shape=[n_hiddens[0], n_hiddens[1]]),
                'out': tf.get_variable("out", shape=[n_hiddens[1], n_classes])
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hiddens[0]])),
                'b2': tf.Variable(tf.random_normal([n_hiddens[1]])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

            # Construct model
            self.prob = self._forward(self.x, self.weights, self.biases)
            self.cost = self._cost(self.prob, self.y, self.a)
            self.pred = tf.argmax(self.prob, 1)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize
        self.init = tf.global_variables_initializer()

        # Session
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

    def _forward(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with softmax activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        out_layer = tf.nn.softmax(out_layer)

        return out_layer

    def _cost(self, prob, y, class_weight):
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=prob,
                                                     targets=y,
                                                     pos_weight=class_weight))
        regularizer = tf.constant(0.)
        for w in self.weights.values():
            regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
        cost = tf.reduce_mean(loss + self.regularization * regularizer)
        return cost

    def fit(self, X, Y, batch_size=None, training_epochs=None,
            display_step=None, imbalanced=None):
        """Fit a training set."""
        if batch_size is None:
            batch_size = self.batch_size
        if training_epochs is None:
            training_epochs = self.training_epochs
        if display_step is None:
            display_step = self.display_step
        if imbalanced is None:
            imbalanced = self.imbalanced
        A = self.class_weight
        # Weighing samples differently based on frequency if imbalanced is True
        if imbalanced is True:
            A *= 1 / (np.sum(Y, axis=0) + 1)
            A /= np.min(A)
            if self.verbose is True:
                print("Using class_weight", A)
        # Preprocessing pipeline
        X = ops.convert_to_tensor(X, dtype="float")
        Y = ops.convert_to_tensor(Y, dtype="float")

        queue_x, queue_y = tf.train.slice_input_producer([X, Y],
                                                         shuffle=True)
        batch_x_op, batch_y_op = tf.train.batch([queue_x, queue_y],
                                                batch_size=batch_size,
                                                num_threads=1)
        # Run init
        self.sess.run(self.init)

        # Start a queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,
                                               coord=coord)

        # Fit
        total_batch = int(int(X.shape[0]) / batch_size)
        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x, batch_y = self.sess.run([batch_x_op, batch_y_op])
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.a: A})
                avg_cost += c / total_batch
                # Display logs per epoch step
            if self.verbose is True and epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(avg_cost))

        coord.request_stop()
        coord.join(threads)

    def predict(self, X):
        """Predict on a test set."""
        pred = self.sess.run(self.pred, feed_dict={self.x: X})
        return pred

    def predict_proba(self, X):
        """Predict probability on a test set."""
        prob = self.sess.run(self.prob, feed_dict={self.x: X})
        return prob

    def close_session(self):
        """Close the running session."""
        self.sess.close()
