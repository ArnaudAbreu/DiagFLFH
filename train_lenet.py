# coding: utf8
#
# Copyright (C) 2019 IUCT-O
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = 'Arnaud Abreu'
__copyright__ = 'Copyright (C) 2019 IUCT-O'
__license__ = 'GNU General Public License'
__version__ = '1.0.0'
__status__ = 'prod'

import os
import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input as preproc_xce
import keras.applications.xception as xce
from keras.layers import Input
from keras.models import Model
import argparse
from keras.optimizers import SGD
import pickle
import errno
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


########################################################################
#
# FUNCTIONS & CLASSES
#
########################################################################
class Brick:

    """
    Neuralizer elementary brick.
    """

    def __init__(self, brickname):

        """
        Arguments:
            - brickname: a string to create the namescope of the brick.
        """

        self.name = brickname
        self.ops = []
        self.trainable_weights = []

    def __str__(self):

        """
        Returns string representation of operations in the brick.
        """

        description = self.name + "\n"
        description += ("-" * len(self.name) + "\n")
        for op in self.ops:
            description += (op.__str__() + "\n")
        return description

    def weights(self):

        """
        Returns all the trainable weights in the brick.
        """

        w = []

        for op in self.ops:
            w += op.trainable_weights

        return w

    def __call__(self, arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=True):
            y = self.ops[0](arg_tensor)
            for op in self.ops[1::]:
                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y


class Classifier(Brick):

    def __init__(self,
                 brickname='classifier',
                 filters=[20, 50],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dropouts=[0., 0.],
                 fc=[500],
                 fcdropouts=[0.],
                 conv_activations=['relu', 'relu'],
                 fc_activations=['relu'],
                 end_activation='softmax',
                 output_channels=10):

        """
        LeNet classical classifier.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = conv_activations[depth]
            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name=poolname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(Flatten())

        for depth in range(len(fc)):

            opname = 'fc_' + str(depth)
            dropname = 'fc_dropout_' + str(depth)
            opunits = fc[depth]
            opdropout = fcdropouts[depth]
            opac = fc_activations[depth]
            self.ops.append(Dense(opunits, activation=opac, name=opname))
            self.ops.append(Dropout(opdropout, name=dropname))

        self.ops.append(Dense(output_channels,
                              activation=end_activation,
                              name='final_fc'))

    def transfer(self, other_clf):

        op_list = []

        for w, wother in zip(self.trainable_weights, other_clf.trainable_weights):

            op_list.append(w.assign(wother))

        return op_list


########################################################################
#
# MAIN
#
########################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="path to png dataset directory.")

    parser.add_argument("--device", default="0",
                        help="ID of the device to use for computation.")

    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs for training the model.")

    parser.add_argument("--batchsize", type=int, default=128,
                        help="number of samples in one batch for fitting.")

    parser.add_argument("--imsize", type=int, default=299, help="size of the side of an image (in pixels)")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # data directories
    DATADIR = args.dataset
    TRAINDIR = os.path.join(DATADIR, 'Training')
    VALIDATIONDIR = os.path.join(DATADIR, 'Validation')
    TESTDIR = os.path.join(DATADIR, 'Test')
    MODELDIR = os.path.join(DATADIR, 'Model')

    # assume every previous directories exists
    for d in [DATADIR, TRAINDIR, VALIDATIONDIR, TESTDIR]:
        if not os.path.exists(d):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), d)

    # create output directory to store the model if does not exists
    modeloutputdir = os.path.join(MODELDIR, 'Lenet')
    if not os.path.exists(modeloutputdir):
        os.makedirs(modeloutputdir)

    # create a data generator with default parameters (no augmentation)
    train_datagen = ImageDataGenerator(preprocessing_function=preproc_xce, horizontal_flip=True, vertical_flip=True)
    valid_datagen = ImageDataGenerator(preprocessing_function=preproc_xce)
    test_datagen = ImageDataGenerator(preprocessing_function=preproc_xce)
    train_generator = train_datagen.flow_from_directory(TRAINDIR, target_size=(args.imsize, args.imsize))
    valid_generator = valid_datagen.flow_from_directory(VALIDATIONDIR, target_size=(args.imsize, args.imsize))
    test_generator = test_datagen.flow_from_directory(TESTDIR, target_size=(args.imsize, args.imsize))

    # datasets dimensions
    trainsteps = int(train_generator.samples / args.batchsize)
    validsteps = int(valid_generator.samples / args.batchsize)
    print("samples train: ", train_generator.samples)
    print("samples validation: ", valid_generator.samples)
    print("train steps: ", trainsteps)
    print("validation steps: ", validsteps)

    # create the model
    base_archi = Classifier(brickname='reference',
                            filters=[32, 64, 128],
                            kernels=[4, 5, 6],
                            strides=[1, 1, 1],
                            dropouts=[0., 0., 0.],
                            fc=[1024, 1024],
                            fcdropouts=[0.5, 0.5],
                            conv_activations=['relu', 'relu', 'relu'],
                            fc_activations=['relu', 'relu'],
                            end_activation='softmax',
                            output_channels=train_generator.num_classes)
    input = Input(shape=(args.imsize, args.imsize, 3))
    predictions = base_archi(input)
    # create final keras model
    model = Model(inputs=input, outputs=predictions)
    # compile model
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # train model
    print('Training model')
    print('#' * 20)
    history = model.fit_generator(train_generator, epochs=args.epochs, validation_data=valid_generator, steps_per_epoch=trainsteps, validation_steps=validsteps)
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(modeloutputdir, "lenet_model.json"), "w") as json_file:
        json_file.write(model_json)
    with open(os.path.join(modeloutputdir, "lenet_history.p"), 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)
    # serialize weights to HDF5
    model.save_weights(os.path.join(modeloutputdir, "lenet_weights.h5"))
    print("Saved model to disk")
