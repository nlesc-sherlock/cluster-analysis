#!/usr/bin/env python
from __future__ import print_function

import os
import sys
from scipy.misc import imsave
import numpy

def convert(filename, dim1, dim2):

    data = numpy.fromfile(filename, '>f').reshape(dim2, dim1)

    outputfile = ".".join(filename.split('.')[0:-1]) + ".png"

    imsave(outputfile, data)


if __name__ == "__main__":
    """ script for converting patterns into png images

    To produce a pattern on DAS5 use for example the following command:
        gpurun ./rungpu.sh -single input_filename output_filename
        gpurun ./rungpu.sh -single /var/scratch/bwn200/Dresden/2560x1920/Praktica_DCZ5.9_1_34178.JPG ./Praktica_DCZ5.9_1_34178.dat

    Be sure to have the application compiled, and the following modules/aliases:
        module load gcc/4.9.3
        module load cuda70
        alias gpurun="srun -N 1 -C TitanX --gres=gpu:1"

    This script also requires Numpy, Scipy and Pillow, which can be installed with the following commands:
        pip install numpy
        pip install scipy
        pip install Pillow

    This script is then called using:
        ./pattern2png.py filename dimension1 dimension2

    """

    if len(sys.argv) != 4:
        print("Usage: ./pattern2png.py filename dimension1 dimension2")
        exit()
    filename = sys.argv[1]
    dim1 = int(sys.argv[2])
    dim2 = int(sys.argv[3])
    if os.path.isfile(filename):
        convert(filename, dim1, dim2)
    else:
        print("Error: no such file.")
        exit()
