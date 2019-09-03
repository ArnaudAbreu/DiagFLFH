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
import pickle
import argparse

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

    parser.add_argument("--data", type=str, help="path to dataset directory.")

    parser.add_argument("--train_ratio", type=float, default=0.50,
                        help="ratio training/total.")

    parser.add_argument("--valid_ratio", type=float, default=0.25,
                        help="ratio validation/total.")

    parser.add_argument("--wsi_type", type=str, default=".mrxs",
                        help="type of wsi files to use.")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed used for randomization processes.")

    args = parser.parse_args()

    classnames = []
    classdirs = []

    # first, get class folders and names
    for name in os.listdir(args.data):
        if name[0] != '.' and os.path.isdir(os.path.join(args.data, name)):
            classnames.append(name)
            classdirs.append(os.path.join(args.data, name))

    # Then get class wsi files basenames
    datasets = {}
    for k in range(len(classdirs)):
        classdir = classdirs[k]
        classname = classnames[k]
        datasets[classname] = []
        for name in os.listdir(classdir):
            if name[0] != '.':
                filename, extension = os.path.splitext(name)
                if extension == args.wsi_type:
                    datasets[classname].append(filename)
        print("found class: ", classname, " with ", len(datasets[classname]), " ", args.wsi_type, " files.")

    # randomize
    randomizer = numpy.random.RandomState(seed=args.random_seed)
    for classname in classnames:
        randomizer.shuffle(datasets[classname])

    # split
    splitted_datasets = {}
    for k in range(len(classdirs)):
        classname = classnames[k]
        classdir = classdirs[k]
        trainsize = int(args.train_ratio * len(datasets[classname]))
        validsize = int(args.valid_ratio * len(datasets[classname]))

        splitted_datasets[classname] = {'training': [os.path.join(classdir, name + args.wsi_type) for name in datasets[classname][0:trainsize]],
                                        'validation': [os.path.join(classdir, name + args.wsi_type) for name in datasets[classname][trainsize:trainsize + validsize]],
                                        'test': [os.path.join(classdir, name + args.wsi_type) for name in datasets[classname][trainsize + validsize::]]}

        print("dataset: ", classname)
        print("training size: ", len(splitted_datasets[classname]["training"]))
        print("validation size: ", len(splitted_datasets[classname]["validation"]))
        print("test size: ", len(splitted_datasets[classname]["test"]))
        print("-" * 20)

    print("writing datasets summary at: ", os.path.join(args.data, 'dataset.p'))

    with open(os.path.join(args.data, 'dataset.p'), 'wb') as f:
        pickle.dump(splitted_datasets, f)
