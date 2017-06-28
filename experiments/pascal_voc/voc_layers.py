import caffe
import json
import numpy as np
from PIL import Image

import random

class VOCSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class BinVOCSegDataLayer(VOCSegDataLayer):
    """
    Obfuscating the class labels for the VOC dataset
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationBinary/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class BinVOCSegDataLayerDrop(VOCSegDataLayer):
    """
    Obfuscating the class labels for the VOC dataset
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationBinary/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        # random drop prob
        if random.random() < 0.5:
            label[(label > 0) & (label < 255)] = 0
        label = label[np.newaxis, ...]
        return label


class RandVOCSegDataLayer(VOCSegDataLayer):
    """
    Randomizing the foreground segments labels
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationRandom/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class CatVOCSegDataLayer(VOCSegDataLayer):
    """
    Categorize the foreground segments labels
    """
    def load_label(self, idx):
        im = Image.open('{}/SegmentationCategory/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


class GrpVOCSegDataLayer(VOCSegDataLayer):
    """
    Group multiple classes to one
    """
    def load_label(self, idx):
        im = Image.open('{}/SegmentationCategory/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        with open('class_transit.json') as json_data:
            dict_transit = json.load(json_data)
        for k, v in dict_transit.items():
            label[label==int(k)] = v
        label = label[np.newaxis, ...]
        return label


class PartialVOCSegDataLayer(VOCSegDataLayer):
    """
    Keep only five classes (16, 17, 18, 19, 20) and the background (0)
    """
    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # ignore the first 15 classes
        self.label = self.ignore_label(self.label)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def ignore_label(self, label):
        """
        Ignore labels that do not belong to the last 5 classes
        """
        if 'pt4' in self.split:
            label[(label > 15) & (label <= 20)] = 0
        elif 'pt3' in self.split:
            label[(label > 10) & (label <= 15)] = 0
            label[(label > 15) & (label <=20)] -= 5
        elif 'pt2' in self.split:
            label[(label > 5) & (label <= 10)] = 0
            label[(label > 10) & (label <= 20)] -= 5
        elif 'pt1' in self.split:
            label[(label > 0) & (label <= 5)] = 0
            label[(label > 5) & (label <= 20)] -= 5
        elif 'ft4' in self.split:
            label[(label <= 15)] = 0
            label[(label > 15) & (label <= 20)] -= 15
        elif 'ft3' in self.split:
            label[(label <= 10) | (label > 15) & (label <= 20)] = 0
            label[(label > 10) & (label <= 15)] -= 10
        elif 'ft2' in self.split:
            label[(label <= 5) | (label > 10) & (label <= 20)] = 0
            label[(label > 5) & (label <= 10)] -= 5
        elif 'ft1' in self.split:
            label[(label > 5) & (label <= 20)] = 0
        return label



class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class IncSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Incomplete segmentation class layer 20% object missing rate
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/inccls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['INCcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class BinSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Obfuscate the class labels for the SBDD dataset
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['BINcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class BinSBDDSegDataLayerDrop(SBDDSegDataLayer):
    """Drop binary segmentation with prob."""
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['BINcls'][0]['Segmentation'][0].astype(np.uint8)
        # random drop prob
        if random.random() < 0.5:
            label[(label > 0) & (label < 255)] = 0
        label = label[np.newaxis, ...]
        return label



class RandSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Randomizing the foreground segments labels
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/randcls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['RANDcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class CatSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Categorizing the foreground segments labels
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['CATcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class InstSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Instansifying the foreground segments labels (each instance is a class)
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/instcls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['INSTcls'][0]['Segmentation'][0].astype(np.uint16)
        label = label[np.newaxis, ...]
        return label


class GrpSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Group every k foreground segments labels of the same class with the same label
    """
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        with open('class_transit.json') as json_data:
            dict_transit = json.load(json_data)
        for k, v in dict_transit.items():
            label[label==int(k)] = v
        label = label[np.newaxis, ...]
        return label


class PartialSBDDSegDataLayer(SBDDSegDataLayer):
    """
    Keep only five classes (16, 17, 18, 19, 20) and the background (0)
    """
    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # ignore the first 15 classes
        self.label = self.ignore_label(self.label)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def ignore_label(self, label):
        """
        Ignore labels that do not belong to the last 5 classes
        """
        if 'pt4' in self.split:
            label[(label > 15) & (label <= 20)] = 0
        elif 'pt3' in self.split:
            label[(label > 10) & (label <= 15)] = 0
            label[(label > 15) & (label <=20)] -= 5
        elif 'pt2' in self.split:
            label[(label > 5) & (label <= 10)] = 0
            label[(label > 10) & (label <= 20)] -= 5
        elif 'pt1' in self.split:
            label[(label > 0) & (label <= 5)] = 0
            label[(label > 5) & (label <= 20)] -= 5
        elif 'ft4' in self.split:
            label[(label <= 15)] = 0
            label[(label > 15) & (label <= 20)] -= 15
        elif 'ft3' in self.split:
            label[(label <= 10) | (label > 15) & (label <= 20)] = 0
            label[(label > 10) & (label <= 15)] -= 10
        elif 'ft2' in self.split:
            label[(label <= 5) | (label > 10) & (label <= 20)] = 0
            label[(label > 5) & (label <= 10)] -= 5
        elif 'ft1' in self.split:
            label[(label > 5) & (label <= 20)] = 0
        return label

