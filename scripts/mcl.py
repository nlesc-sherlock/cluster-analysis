#!/usr/bin/env python
#
# A wrapper around mcl command-line tool to analyse data using a series of
# input parameters (inflation values).
#
# Note:
#   First, install the MCL package (http://micans.org/mcl/)
#   OS X  : port install mcl 
#   Ubuntu: apt-get install mcl 
#

import os, sys
import argparse as argp
import numpy as np


parser = argp.ArgumentParser(
   description = 'Clustering using Markov Cluster Algorithm (MCL).')

parser.add_argument(
   '-i',
   dest     = 'inflation',
   required = True,
   help     = 'inflation parameter string in the form #start:#end:#increment')

parser.add_argument(
   'infile',
   help = 'input (graph) file in the form of an edge list')

args = parser.parse_args()
infile = args.infile
inflation = args.inflation
start, end, inc = None, None, None

if os.path.isfile(infile) is False:
   parser.error("Input file '%s' not found" % infile)

try:
   (start, end, inc) = inflation.split(':')
   start = float(start)
   end = float(end)
   inc = float(inc)
except:
   parser.print_help()
   sys.exit(1)

for i in np.arange(start, end, inc):
   outfile = infile + '.mcl_I_' + str(i) + '.cls'
   MCL_CMD = 'mcl %s --abc -I %f -o %s' % (infile, i, outfile)
   os.system(MCL_CMD)
