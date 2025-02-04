"""Train a simple deep CNN on the CIFAR10 small images dataset.

From fchollet/keras/keras/examples/cifar10_cnn.py
"""

from __future__ import print_function
import keras
import requests
import argparse
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)
from keras.models import load_model
from models import CNN, VGG8
from wide_resnet import WideResidualNetwork

parser = argparse.ArgumentParser(description='Continue a training.')
parser.add_argument('-t', help='The title of the training to continue')
parser.add_argument('--net', default='vgg',
                    help='Model to use: cnn, vgg, or wide_resnet')
parser.add_argument('--pct_missing', default=0.,
                    help='Percentage of mising "positives"')
parser.add_argument('--asym', help='Class weight of class 0.')
parser.add_argument('--trans', help='Output translate of class 0.')
parser.add_argument('--checkpoint', help='Checkpoint file.')
args = parser.parse_args()

if args.t:
    title = args.t
else:
    r = requests.get('https://frightanic.com/goodies_content/docker-names.php')
    if r.raise_for_status():
        raise
    title = r.text.rstrip()
pct_missing = float(args.pct_missing)
net = args.net
batch_size = 64
num_classes = 11  # 0: negatives and 1-10: cifar10 classes
epochs = 100
data_augmentation = True
class_weight = {k: 1 for k in range(num_classes)}
if args.asym:
    class_weight[0] = float(args.asym)
print("Class weights:", class_weight)
t_vec = np.zeros(num_classes)
if args.trans:
    t_vec[0] = float(args.trans)
print("Translate vector:", t_vec)


# Import cifar10 data as P set
(px_train, py_train), (px_test, py_test) = cifar10.load_data()
print('px_train shape:', px_train.shape)
print('px_test.shape:', px_test.shape)
# Shift py from 0-9 to 1-10
py_train += 1
py_test += 1
assert 0 not in np.unique(py_train)
assert 0 not in np.unique(py_test)
# Import cifar100 data as U set
(ux_train, uy_train), (ux_test, uy_test) = cifar100.load_data()
print('ux_train shape:', ux_train.shape)
print('ux_test.shape:', ux_test.shape)
# Mask all classes from cifar10 as 0 (negatives)
uy_train[...] = 0
uy_test[...] = 0
assert uy_train.shape == (50000, 1)
assert np.unique(uy_train) == [0]
# Combine P set and U set
x_train = np.concatenate((px_train, ux_train), axis=0).astype('float32')
y_train = np.concatenate((py_train, uy_train), axis=0).astype('int8')
x_test = np.concatenate((px_test, ux_test), axis=0).astype('float32')
y_test = np.concatenate((py_test, uy_test), axis=0).astype('int8')
# Construct artificial U set
unique, counts = np.unique(y_train, return_counts=True)
print('Complete labels: ', {k: v for k, v in zip(unique, counts)})
np.random.seed(42)
num_samples = y_train.shape[0]
missing = np.random.rand(num_samples) < pct_missing
for idx in range(num_samples):
    if y_train[idx] > 0 and missing[idx]:
        y_train[idx] = 0
unique, counts = np.unique(y_train, return_counts=True)
print("Missing {} positives.".format(pct_missing))
print('Missing labels: ', {k: v for k, v in zip(unique, counts)})
np.random.seed(None)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


opt = 'adam'
if net.lower() == 'cnn':
    model = CNN(input_shape=x_train.shape[1:], num_classes=num_classes)
    # initiate RMSprop optimizer
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
elif net.lower() == 'vgg':
    model = VGG8(input_shape=x_train.shape[1:],
                 num_classes=num_classes, t_vec=t_vec)
else:
    model = WideResidualNetwork(depth=28, width=8, dropout_rate=0.5,
                                classes=11, include_top=True, weights=None)

if args.checkpoint:
    model = load_model(args.checkpoint)

model.summary()


# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def normalize(x):
    """Substract mean and Divide by std."""
    x -= np.array([125.3, 123.0, 113.9])
    x /= np.array([63.0, 62.1, 66.7])
    return x


x_train = normalize(x_train)
x_test = normalize(x_test)

# Checkpoint
checkpointer = ModelCheckpoint(
    filepath="model_checkpoint_{}_{}.h5".format(pct_missing, title),
    verbose=1,
    save_best_only=True)

# csvlogger
csv_logger = CSVLogger(
    'csv_logger_{}_{}.csv'.format(pct_missing, title))
# EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=20)
# Reduce lr on plateau
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10,
                               min_lr=0.5e-5)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[csv_logger, checkpointer, early_stopper])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        # divide inputs by std of the dataset
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        class_weight=class_weight,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[csv_logger, checkpointer, early_stopper])
