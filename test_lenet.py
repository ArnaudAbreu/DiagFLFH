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
from keras.models import model_from_json
import argparse
import pickle
import keras.backend as K
from skimage.io import imread

########################################################################
#
# FUNCTIONS & CLASSES
#
########################################################################


########################################################################
#
# MAIN
#
########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, help="path to png data directory.")

    parser.add_argument("--device", default="0",
                        help="ID of the device to use for computation.")

    parser.add_argument("--batchsize", type=int, default=128,
                        help="number of samples in one batch for fitting.")

    parser.add_argument("--mcsampling", type=int, default=30,
                        help="sampling of the mc posterior.")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # data directories
    DATADIR = args.datadir
    TRAINDIR = os.path.join(DATADIR, 'Training')
    VALIDATIONDIR = os.path.join(DATADIR, 'Validation')
    TESTDIR = os.path.join(DATADIR, 'Test')
    MODELDIR = os.path.join(DATADIR, 'Model')

    # output directory where model is stored
    modeloutputdir = os.path.join(MODELDIR, 'Lenet')

    # assume every previous directories exists
    for d in [DATADIR, TRAINDIR, VALIDATIONDIR, TESTDIR, MODELDIR, modeloutputdir]:
        if not os.path.exists(d):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), d)

    # prediction output directory
    testoutputdir = os.path.join(DATADIR, 'TestOutputLenet')
    os.makedirs(testoutputdir)

    # get classnames (suffixes)
    suffixes = [name for name in os.listdir(TESTDIR) if name[0] != '.' and os.path.isdir(os.path.join(TESTDIR, name))]

    # create a data generator with default parameters (no augmentation)`
    def images():
        for sufix in suffixes:
            infolder = os.path.join(TESTDIR, sufix)
            for name in os.listdir(infolder):
                # name is a png filename
                if name[0] != '.' and '.png' in name:
                    impath = os.path.join(infolder, name)
                    base = name[0:-len('.png')]
                    # yield basename of the file and image as numpy array
                    yield base, imread(impath)

    def batches():
        # batches of images
        # size is args.batchsize
        batch = []
        names = []
        for name, image in images():
            if len(batch) == args.batchsize:
                batch = []
                names = []
            batch.append(image)
            names.append(name)
            if len(batch) == args.batchsize:
                yield names, batch
        # yield potential remaining batch
        if len(batch):
            yield names, batch

    # CODE TO RELOAD THE MODEL

    # load json and create model
    json_file = open(os.path.join(modeloutputdir, "lenet_model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(modeloutputdir, "lenet_weights.h5"))
    print("Loaded model from disk")

    # create a function to make MC dropout predictions
    predictor = K.function([loaded_model.layers[0].input, K.learning_phase()], [loaded_model.layers[-1].output])

    def mc_dropout_predict(predfunc, x, sampling=args.mcsampling):
        result = numpy.zeros((sampling,) + (x.shape[0], 2))

        for i in range(sampling):
            result[i, :, :] = predfunc((x, 1))[0]

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    count = 0

    for batch in batches():
        names, ims = batch
        inputs = preproc_xce(numpy.array(ims))
        predictions, stds = mc_dropout_predict(predictor, inputs)
        outpath = os.path.join(testoutputdir, 'batch' + str(count) + '.p')
        with open(outpath, 'wb') as f:
            # every batch prediction is stored in a pickle file
            pickle.dump({name: (pred, std) for name, pred, std in zip(names, predictions, stds)}, f)
        count += 1
        print(count * args.batchsize, ' images processed!')
