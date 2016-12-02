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
    if len(sys.argv) != 4:
        print("Usage: ./pattern2png.py filename dimension1 dimension2.")
        exit()
    filename = sys.argv[1]
    dim1 = int(sys.argv[2])
    dim2 = int(sys.argv[3])
    if os.path.isfile(filename):
        convert(filename, dim1, dim2)
    else:
        print("Error: no such file.")
        exit()
