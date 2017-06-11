"""A multilayer perceptron implemented with TensorFlow."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class MultilayerPerceptron(object):
    """MultilayerPerceptron.

    Parameters
    ------------
    n_classes : int
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
    learning_rate : float
        Learning rate (between 0.0 and 1.0). Defaults to 0.01
    epochs : int
        The number of passes over the training data. Defaults to 100.
    alpha : float
        Constant that multiplies the L2 regularization term. Defaults to 0.0.
    batch_size: int
        Size of minibatches for stochastic optimizers. Defaults to 200.
    class_weight: dict, {class_label: weight} or None, optional
        Preset for the class_weight fit parameter.
        Weights associated with classes. If not given, all classes are
        supposed to have weight one.
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.

    """

    def __init__(self, n_input, n_classes, n_hiddens=[8, 8],
                 learning_rate=0.01,
                 batch_size=100,
                 epochs=30, alpha=1e-3,
                 display_step=10,
                 class_weight=None,
                 verbose=None):
        """Init."""
        # Fit Parameters
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hiddens = n_hiddens
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = alpha
        self.display_step = display_step
        self.class_weight = class_weight
        self.verbose = verbose

        self._build()

    def _build(self):
        """Build the Graph."""
        # Define Graph
        tf.reset_default_graph()

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # loss class weight
        self.a = tf.placeholder("float", [self.n_classes])

        # Layer weights and biases
        self.weights = {
            'h1': tf.get_variable(
                "h1", shape=[self.n_input, self.n_hiddens[0]]),
            'h2': tf.get_variable(
                "h2", shape=[self.n_hiddens[0], self.n_hiddens[1]]),
            'out': tf.get_variable(
                "out", shape=[self.n_hiddens[1], self.n_classes])
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hiddens[0]])),
            'b2': tf.Variable(tf.random_normal([self.n_hiddens[1]])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        self.out = self._forward(self.x, self.weights, self.biases)
        self.loss = self._loss(self.out, self.y, self.a)
        self.cost = self._regularize(self.loss)
        self.prob = tf.nn.softmax(self.out)
        self.pred = tf.argmax(self.prob, 1)
        self.out_grad = tf.gradients(self.cost, [self.out])

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize
        self.init = tf.global_variables_initializer()

        # Session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
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

        return out_layer

    def _loss(self, out, y, class_weight):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=out,
            targets=y,
            pos_weight=class_weight)
        loss = tf.reduce_sum(loss, axis=1)

        return loss

    def _regularize(self, loss):
        loss = tf.reduce_mean(loss)
        regularizer = tf.constant(0.)
        for w in self.weights.values():
            regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
        cost = tf.reduce_mean(loss + self.alpha * regularizer)
        return cost

    def fit(self, X, Y):
        """Fit a training set."""
        batch_size = self.batch_size
        epochs = self.epochs
        display_step = self.display_step
        class_weight = self._to_class_weight(Y)

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
        for epoch in range(epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x, batch_y = self.sess.run([batch_x_op, batch_y_op])
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.a: class_weight})
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

    def calc_loss(self, X, Y):
        """Return loss."""
        class_weight = self._to_class_weight(Y)

        loss = self.sess.run(self.loss, feed_dict={self.x: X,
                                                   self.y: Y,
                                                   self.a: class_weight})
        return loss

    def calc_gradient(self, X, Y, imbalanced=None):
        """Return loss gradient w.r.t the last layer output."""
        class_weight = self._to_class_weight(Y)

        grad = self.sess.run(self.out_grad, feed_dict={self.x: X,
                                                       self.y: Y,
                                                       self.a: class_weight})
        return grad[0]

    def _to_class_weight(self, Y):
        n_samples, n_classes = Y.shape
        if self.class_weight is None:
            class_weight = np.ones(n_classes)
        elif self.class_weight is "balanced":
            class_weight = np.ones(n_classes) / (np.sum(Y, axis=0) + 1)
        else:
            classes = sorted(self.class_weight.keys())
            if len(classes) != n_classes:
                raise ValueError("The length of class_weight has to "
                                 "match to n_classes.")
            class_weight = np.asarray([self.class_weight[k] for k in classes])
        class_weight = class_weight.astype('float')
        class_weight = class_weight / np.min(class_weight)
        if self.verbose:
            print("Using {}".format(class_weight))
        return class_weight

    def close_session(self):
        """Close the running session."""
        self.sess.close()
