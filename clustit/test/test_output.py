from __future__ import print_function

from .context import clustit
import numpy
import clustit.embedding as embedding
from clustit.output import OutputCollection

import json

import os
test_files = os.path.dirname(os.path.realpath(__file__)) + "/test_files/"


def test_output():
    df = embedding.largevis(test_files + 'edgelist-pentax-pce.txt', outdim=3, alpha=0.1)

    oc = OutputCollection(df)
    print(oc)

    n = df.filename.size
    values = numpy.random.randn(n)
    print(values)
    oc.add_property("random_number1", values)
    values = numpy.random.randn(n)
    oc.add_property("random_number2", values)

    json_string = oc.to_DiVE()
    print(json_string)

    try:
        json_object = json.loads(json_string)
    except ValueError as e:
        assert False


    assert False

    assert True


def test_output2():
    df = embedding.largevis(test_files + 'edgelist-pentax-pce.txt', outdim=3, alpha=0.1)
    oc = OutputCollection(df)

    array = oc.to_array()
    assert array.shape == (5,3)

