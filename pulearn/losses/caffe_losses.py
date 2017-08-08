"""Python implementation of Caffe SoftmaxWithLossLayer."""
import caffe
import numpy as np


class SoftmaxWithLossLayer(caffe.Layer):
    """Pycaffe Layer for SoftmaxWithLoss."""

    def setup(self, bottom, top):
        r"""Setup the layer with params.

        Example params:
        layer {
          name: "loss"
          type: "Python"
          bottom: "score"
          bottom: "label"
          top: "loss"
          loss_weight: 1
          python_param {
            module: "pyloss"
            layer: "SoftmaxWithLossLayer"
            param_str: "{\'ignore_label\': 255, \'loss_weight\': 1, "
                       "\'normalization\': 1, \'axis\': 1}"
          }
        }
        """
        # config: python param
        self.params = eval(self.param_str)
        # softmax_param
        self._softmax_axis = self.params.get('axis', 1)
        self._num_output = bottom[0].shape[self._softmax_axis]
        # loss_param
        self._normalization = self.params.get('normalization', 2)
        self._ignore_label = self.params.get('ignore_label', None)
        self._loss_weight = self.params.get('loss_weight', 1)
        # attributes initialization
        self.loss = None
        self.prob = None
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute softmax loss.")

    def reshape(self, bottom, top):
        """Reshape the layer."""
        # check input dimensions match
        if bottom[0].count != bottom[1].count * bottom[0].channels:
            raise Exception("Number of labels must match "
                            "number of predictions; "
                            "e.g., if softmax axis == 1 "
                            "and prediction shape is (N, C, H, W), "
                            "label count (number of labels) "
                            "must be N*H*W, with integer values "
                            "in {0, 1, ..., C-1}.")
        # loss output is scalar
        top[0].reshape(1)
        # softmax output is of same shape as bottom[0]
        if len(top) >= 2:
            top[1].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        """Forward computing on CPU."""
        # compute stable softmax probability
        score = bottom[0].data
        score -= np.max(score, axis=self._softmax_axis, keepdims=True)
        score_exp = np.exp(score)
        prob = score_exp / np.sum(score_exp, axis=self._softmax_axis,
                                  keepdims=True)
        # compute negative log-likelihood loss
        label = bottom[1].data.astype('int8')
        loss = self.compute_loss(prob, label)
        # sum up loss and normalize
        loss = np.sum(loss) / float(self.get_normalizer(label))
        # pass loss top[0]
        top[0].data[...] = loss
        # update loss and prob
        self.loss = loss
        self.prob = prob

    def backward(self, top, propagate_down, bottom):
        """Backward computing on CPU."""
        if propagate_down[1]:
            raise Exception("SoftmaxWithLoss Layer cannot "
                            "backpropagate to label inputs.")
        if propagate_down[0]:
            label = bottom[1].data.astype('int8')
            bottom_diff = self.compute_diff(self.prob, label)
            # normlize diff
            loss_weight = self._loss_weight / float(self.get_normalizer(label))
            bottom_diff = loss_weight * bottom_diff
            # pass the derivatives to bottom[0]
            bottom[0].diff[...] = bottom_diff

    def compute_loss(self, prob, label):
        """Return softmax loss."""
        # ignore label == ignore_label
        ignore_mask = self.ignore_mask(label, tiled=True)
        label[label == self._ignore_label] = 0
        # negative logarithm of prob
        neg_log = -np.log(prob.clip(min=1e-16))
        # if loss_param has ignore_label
        if self._ignore_label:
            neg_log[ignore_mask] = 0
        # compute loss $ l_i = - y_ik \ln a_ik, where i:-sample, k:-class $
        label_1hot = self.one_hot_encode(label, squeeze=True)
        loss = label_1hot * neg_log
        return loss

    def compute_diff(self, prob, label):
        """Return sofmax loss derivative."""
        # ignore label == ignore_label
        ignore_mask = self.ignore_mask(label, tiled=True)
        label[label == self._ignore_label] = 0
        # compute derivative $ dldj = a_ij - y_ij$
        label_1hot = self.one_hot_encode(label, squeeze=True)
        diff = prob - label_1hot
        # apply ignore mask for label == ignore_label
        diff[ignore_mask] = 0
        return diff

    def get_normalizer(self, label):
        """Get the loss normalizer based normalization mode."""
        if self._normalization == 0:    # Full
            normalizer = label.size
        elif self._normalization == 1:  # VALID
            normalizer = np.sum(label != self._ignore_label)
        elif self._normalization == 2:  # BATCH_SIZE
            normalizer = label.shape[0]
        elif self._normalization == 3:  # NONE
            normalizer = 1.
        else:
            raise Exception("Unknown normalization mode: {}").format(
                self._normalization)
        return max(1., normalizer)

    def reduce_prob(self, prob, label):
        """Return probabilities for given labels."""
        indices = np.indices(label.shape)
        indices[self._softmax_axis] = label
        return prob[tuple(indices)]

    def one_hot_encode(self, label, squeeze=True):
        """Return one hot encoded labels."""
        if squeeze:
            label_sqz = np.squeeze(label, (self._softmax_axis,))
        else:
            label_sqz = label
        label_1hot = np.eye(self._num_output)[label_sqz]
        label_1hot = np.rollaxis(label_1hot, -1, self._softmax_axis)
        return label_1hot

    def ignore_mask(self, label, tiled=False):
        """Return label ignore mask."""
        if tiled is True:
            repeats = [1] * len(label.shape)
            repeats[self._softmax_axis] = self._num_output
            return np.tile(label == self._ignore_label, repeats)
        else:
            return label == self._ignore_label


class WeightedSoftmaxWithLossLayer(SoftmaxWithLossLayer):
    """An asymmetric softmax loss weighted by class dependent weights."""

    def setup(self, bottom, top):
        """Parse class-dependent weight from params."""
        super(WeightedSoftmaxWithLossLayer, self).setup(bottom, top)
        self.class_weight = self.params.get(
            'class_weight', np.ones(self._num_output))
        self.class_weight = self._to_class_weight(self.class_weight)

    def compute_loss(self, prob, label):
        """Return softmax loss."""
        loss = super(WeightedSoftmaxWithLossLayer, self).compute_loss(prob,
                                                                      label)
        # weigh loss with class dependent betas $l_i = - \beta_i y_ik \ln a_ik$
        loss = self._to_sample_weight(label) * loss
        return loss

    def compute_diff(self, prob, label):
        """Return softmax loss derivative."""
        diff = super(WeightedSoftmaxWithLossLayer, self).compute_diff(prob,
                                                                      label)
        # weigh diff with class dependent betas $dldj = \beta_i (a_ij - y_ij)$
        diff = self._to_sample_weight(label) * diff
        return diff

    def _to_class_weight(self, class_weight):
        if class_weight is None:
            class_weight = np.ones(self._num_output)
        else:
            classes = sorted(class_weight.keys())
            if len(classes) != self._num_output:
                raise ValueError("The length of class_weight has to "
                                 "match to n_classes.")
            class_weight = np.asarray([self.class_weight[k] for k in classes])
        class_weight = class_weight.astype('float')
        # class_weight = class_weight / np.min(class_weight)
        return class_weight

    def _to_sample_weight(self, label, tiled=True):
        sample_weight = self.class_weight[label]
        repeats = [1] * len(label.shape)
        repeats[self._softmax_axis] = self._num_output
        return np.tile(sample_weight, repeats)


class ExponentialWithLossLayer(WeightedSoftmaxWithLossLayer):
    def setup(self, bottom, top):
        """Parse class-dependent weight from params."""
        super(ExponentialWithLossLayer, self).setup(bottom, top)
        self.class_weight = self.params.get(
            'class_weight', np.ones(self._num_output))
        self.class_weight = self._to_class_weight(self.class_weight)

    def compute_loss(self, prob, label):
        ignore_mask = self.ignore_mask(label, tiled=True)
        loss = super(ExponentialWithLossLayer, self).compute_loss(prob, label)
        negmask = self.negative_mask(label, tiled=True)
        negative_loss = np.zeros(loss.shape)
        negative_loss[negmask] = 1 - prob[negmask]
        negative_loss[:, 1:, ...] = 0
        loss[negmask] = negative_loss[negmask]
        if self._ignore_label:
            loss[ignore_mask] = 0
        return loss

    def compute_diff(self, prob, label):
        ignore_mask = self.ignore_mask(label, tiled=True)
        diff = super(ExponentialWithLossLayer, self).compute_diff(prob, label)
        negmask = self.negative_mask(label, tiled=True)
        negative_diff = np.zeros(diff.shape)
        # cls = 0
        negmask_0 = negmask.copy()
        negmask_0[:, 1:, ...] = False
        negative_diff[negmask_0] = prob[negmask_0] * (prob[negmask_0] - 1)
        # cls != 0
        for cls in range(1, self._num_output):
            negmask_1 = negmask.copy()
            negmask_1[:, :cls, ...] = False
            negmask_1[:, cls + 1:, ...] = False
            negative_diff[negmask_1] = prob[negmask_0] * prob[negmask_1]
        diff[negmask] = negative_diff[negmask]
        if self._ignore_label:
            diff[ignore_mask] = 0
        return diff

    def negative_mask(self, label, tiled=True):
        if tiled is True:
            repeats = [1] * len(label.shape)
            repeats[self._softmax_axis] = self._num_output
            return np.tile(label == 0, repeats)
        else:
            return label == 0


class HardBootstrappingSoftmaxWithLossLayer(WeightedSoftmaxWithLossLayer):
    """A softmax loss introduce consistency Hard bootstrapping."""

    def compute_loss(self, prob, label):
        """Return softmax loss."""
        # ignore label == ignore_label
        ignore_mask = self.ignore_mask(label, tiled=True)
        label[label == self._ignore_label] = 0
        # negative logarithm of prob
        neg_log = -np.log(prob.clip(min=1e-16))
        # if loss_param has ignore_label
        if self._ignore_label:
            neg_log[ignore_mask] = 0
        # compute loss $l_i = - (\beta_i y_ik+(1-\beta_i) \hat{y_ik}) \ln a_ik$
        pred = np.argmax(prob, axis=self._softmax_axis)
        pred_1hot = self.one_hot_encode(pred, squeeze=False)
        label_1hot = self.one_hot_encode(label, squeeze=True)
        betas = self.get_sample_betas(label)
        loss = (betas * label_1hot + (1 - betas) * pred_1hot) * neg_log
        return loss

    def compute_diff(self, prob, label):
        """Return softmax loss derivative."""
        # ignore label == ignore_label
        ignore_mask = self.ignore_mask(label, tiled=True)
        label[label == self._ignore_label] = 0
        # compute diff $dldj=\beta_i (a_ij-y_ij)+(1-\beta_i)(a_ij-\hat{y_ij})$
        pred = np.argmax(prob, axis=self._softmax_axis)
        pred_1hot = self.one_hot_encode(pred, squeeze=False)
        label_1hot = self.one_hot_encode(label, squeeze=True)
        betas = self.get_sample_betas(label)
        diff = betas * (prob - label_1hot) + (1 - betas) * (prob - pred_1hot)
        # apply ignore mask for label == ignore_label
        diff[ignore_mask] = 0
        return diff
