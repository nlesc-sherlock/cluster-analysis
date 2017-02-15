from __future__ import print_function

#from .context import clustit
import numpy
from nose.tools import raises
from clustit import embedding

import os
test_files = os.path.dirname(os.path.realpath(__file__)) + "/test_files/"

#test_files=os.getcwd()+"/test/test_files"

def test_largevis():
    output = embedding.largevis(test_files + 'edgelist.txt', outdim=2, alpha=0.1)

    print(output)
    print(output.shape)

    labels = [str(s) for s in output.filename[:]]
    expected = ['node001', 'node002', 'node003']

    assert len(labels) == len(expected)
    assert all([a in labels for a in expected])
    assert output.shape == (3,3)



if __name__ == "__main__":
    test_largevis()

