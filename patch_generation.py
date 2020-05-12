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
from openslide import OpenSlide
from skimage.io import imsave
from tqdm import tqdm
from skimage.exposure import is_low_contrast
import itertools


########################################################################
#
# FUNCTIONS & CLASSES
#
########################################################################
def get_tissue(image, blacktol=0, whitetol=230):
    """
    Given an image and a tolerance on black and white pixels,
    returns the corresponding tissue mask segmentation, i.e. true pixels
    for the tissue, false pixels for the background.

    Arguments:
        - image: numpy ndarray, rgb image.
        - blacktol: float or int, tolerance value for black pixels.
        - whitetol: float or int, tolerance value for white pixels.

    Returns:
        - binarymask: true pixels are tissue, false are background.
    """

    binarymask = numpy.ones_like(image[:, :, 0], bool)

    for color in range(3):
        # for all color channel, find extreme values corresponding to black or white pixels
        binarymask = numpy.logical_and(binarymask, image[:, :, color] < whitetol)
        binarymask = numpy.logical_and(binarymask, image[:, :, color] > blacktol)

    return binarymask


def randomized_regular_seed(shape, width, randomizer):
    maxi = width * int(shape[0] / width)
    maxj = width * int(shape[1] / width)
    col = numpy.arange(start=0, stop=maxj, step=width, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=width, dtype=int)
    randomizer.shuffle(col)
    randomizer.shuffle(line)
    for p in itertools.product(line, col):
        yield p


def randomized_patches(slide, level, patchsize, randomizer, n_max):
    counter = 0
    shape = slide.level_dimensions[level]
    for i, j in randomized_regular_seed(shape, patchsize, randomizer):
        # check number of images already yielded
        if counter >= n_max:
            break
        # get the image
        x = j * (2 ** level)
        y = i * (2 ** level)
        image = numpy.array(slide.read_region((x, y), level, (patchsize, patchsize)))[:, :, 0:3]
        # check whether image is yieldable
        if (not is_low_contrast(image)) and (get_tissue(image).sum() > 0.5 * patchsize * patchsize):
            counter += 1
            yield x, y, level, image


########################################################################
#
# MAIN
#
########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasetfile", type=str, help="path to dataset file.")

    parser.add_argument("--outputdir", type=str, help="directory where to put the patches.")

    parser.add_argument("--level", type=int, default=3, help="resolution pyramid level to extract patches.")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed used for randomization processes.")

    parser.add_argument("--maxpatches", type=int, default=100,
                        help="maximum number of patches to extract from a slide.")

    parser.add_argument("--imsize", type=int, default=299, help="size of the side of an image (in pixels)")

    args = parser.parse_args()

    # first get datasets
    with open(args.datasetfile, "rb") as f:
        datasets = pickle.load(f)

    # then create data directories and put patches inside
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # randomizer
    randomizer = numpy.random.RandomState(seed=args.random_seed)

    for subset in ["training", "validation", "test"]:
        # create corresponding directory
        subsetdir = os.path.join(args.outputdir, subset)
        if not os.path.exists(subsetdir):
            os.makedirs(subsetdir)

        print("Generate ", subset, " patches.")
        print("-" * 20)
        # loop over classes
        for classname in datasets:
            patchfolder = os.path.join(subsetdir, classname)
            if not os.path.exists(patchfolder):
                os.makedirs(patchfolder)
            for slidepath in tqdm(datasets[classname][subset]):
                slide = OpenSlide(slidepath)
                slidename, _ = os.path.splitext(os.path.basename(slidepath))
                for x, y, level, image in randomized_patches(slide, args.level, args.imsize, randomizer, args.maxpatches):
                    outpath = os.path.join(patchfolder, slidename + '_' + str(x) + '_' + str(y) + '_' + str(level) + '.png')
                    imsave(outpath, image)
